#!/usr/bin/env python 
import os, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertConfig, BertModel, AutoTokenizer
import numpy as np
import tensorflow as tf  # only for loading TF checkpoints

# ---------------------------------------------------
# Utility
# ---------------------------------------------------
def masked_softmax(scores, mask, dim=-1):
    mask = mask.float()
    scores = scores.masked_fill(~mask.bool(), float('-inf'))
    w = F.softmax(scores, dim=dim) * mask
    return w / (w.sum(dim=dim, keepdim=True) + 1e-13)

def extract_span_tokens(token_embs, spans, max_span_width=30):
    B,L,H = token_embs.size()
    N    = spans.size(1)
    device = token_embs.device

    s = spans[:,:,0].clamp(0, L-1)
    e = spans[:,:,1].clamp(0, L-1)
    widths = (e - s + 1).clamp(1, max_span_width)

    idx      = torch.arange(max_span_width, device=device).view(1,1,-1)
    span_idx = (s.unsqueeze(-1) + idx).clamp(0, L-1)
    mask     = idx < widths.unsqueeze(-1)

    flat_pos   = span_idx.view(-1)
    flat_batch = torch.arange(B, device=device) \
                      .unsqueeze(1).unsqueeze(2) \
                      .expand(-1, N, max_span_width) \
                      .reshape(-1)

    toks  = token_embs[flat_batch, flat_pos].view(B*N, max_span_width, H)
    masks = mask.view(B*N, max_span_width)
    return toks, masks

class HeadAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
    def forward(self, toks, mask):
        scores = self.proj(toks).squeeze(-1)
        weights = masked_softmax(scores, mask)
        return (weights.unsqueeze(-1) * toks).sum(1)

# ---------------------------------------------------
# TF checkpoint loading (unchanged)
# ---------------------------------------------------
_tensors_to_transpose = (
    "dense/kernel","attention/self/query/kernel",
    "attention/self/key/kernel","attention/self/value/kernel",
    "/kernel"
)
_tf_to_pt_map = (
    ('bert/',''),('layer_','layer.'),
    ('word_embeddings','embeddings.word_embeddings.weight'),
    ('position_embeddings','embeddings.position_embeddings.weight'),
    ('token_type_embeddings','embeddings.token_type_embeddings.weight'),
    ('LayerNorm/gamma','LayerNorm.weight'),
    ('LayerNorm/beta','LayerNorm.bias'),
    ('kernel','weight'),('/','.')
)
def _tf_to_pt_var(name):
    for t,p in _tf_to_pt_map:
        name = name.replace(t,p)
    return name

def load_spanbert_backbone_from_tf(path, cfg_path=None, device='cpu'):
    if not os.path.exists(path + ".index"):
        raise FileNotFoundError(f"{path}.index not found")
    var_list = tf.train.list_variables(path)
    cfg = BertConfig.from_json_file(cfg_path) if cfg_path else BertConfig()
    model = BertModel(cfg)
    state = model.state_dict()
    reader = tf.train.load_checkpoint(path)
    for tf_name,_ in var_list:
        if not tf_name.startswith("bert/"): continue
        arr = reader.get_tensor(tf_name)
        pt_name = _tf_to_pt_var(tf_name)
        if pt_name in state:
            if any(x in tf_name for x in _tensors_to_transpose):
                arr = arr.transpose()
            t = torch.from_numpy(arr)
            if t.shape == state[pt_name].shape:
                state[pt_name].copy_(t)
    model.load_state_dict(state)
    return model.to(device)

def _to_coref_key(tf_name):
    return {
        'mention_scores/hidden_weights_0':'mention_ffnn.0.weight',
        'mention_scores/hidden_bias_0':'mention_ffnn.0.bias',
        'mention_scores/output_weights':'mention_ffnn.3.weight',
        'mention_scores/output_bias':'mention_ffnn.3.bias',
        'span_width_prior_embeddings':'width_prior.weight',
        'width_scores/hidden_weights_0':'width_ffnn.0.weight',
        'width_scores/hidden_bias_0':'width_ffnn.0.bias',
        'width_scores/output_weights':'width_ffnn.2.weight',
        'width_scores/output_bias':'width_ffnn.2.bias',
        'mention_word_attn/output_weights':'head_attn.proj.weight',
        'mention_word_attn/output_bias':'head_attn.proj.bias',
        'coref_layer/f/output_weights':'coarse_proj1.weight',
        'coref_layer/f/output_bias':'coarse_proj1.bias',
        'src_projection/output_weights':'coarse_proj2.weight',
        'src_projection/output_bias':'coarse_proj2.bias',
        'coref_layer/slow_antecedent_scores/hidden_weights_0':'ant_ffnn.0.weight',
        'coref_layer/slow_antecedent_scores/hidden_bias_0':'ant_ffnn.0.bias',
        'coref_layer/slow_antecedent_scores/output_weights':'ant_ffnn.3.weight',
        'coref_layer/slow_antecedent_scores/output_bias':'ant_ffnn.3.bias',
        'span_width_embeddings':'width_emb.weight',
        'coref_layer/same_speaker_emb':'speaker_emb.weight',
        'coref_layer/segment_distance/segment_distance_embeddings':'segment_emb.weight',
        'genre_embeddings':'genre_emb.weight',
        'antecedent_distance_emb':'dist_emb.weight'
    }.get(tf_name)

def load_coref_head_from_tf(model, path):
    reader = tf.train.load_checkpoint(path)
    tf_vars = reader.get_variable_to_shape_map()
    state = model.state_dict()
    for tf_key in tf_vars:
        pt = _to_coref_key(tf_key)
        if pt and pt in state:
            arr = reader.get_tensor(tf_key)
            t   = torch.from_numpy(arr)
            exp = state[pt].shape
            if t.dim()==2 and t.shape[::-1]==tuple(exp):
                t = t.transpose(0,1)
            if tuple(t.shape)==tuple(exp):
                state[pt].copy_(t)
    model.load_state_dict(state)
    return model

# ---------------------------------------------------
# SpanBERTFullCoref
# ---------------------------------------------------
class SpanBERTFullCoref(nn.Module):
    def __init__(self, config=None, **kw):
        super().__init__()
        self.bert = BertModel(config or BertConfig())
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.max_span_width = kw.get("max_span_width", 30)
        self.top_span_ratio = kw.get("top_span_ratio", 0.4)
        self.max_top_ant    = kw.get("max_top_ant", 50)
        self.ant_temp       = kw.get("ant_temp", 0.5)
        self.dummy_bias     = nn.Parameter(torch.tensor(-1.0))
        self.link_threshold = kw.get("link_threshold", 0.5)
        self.fine_grained   = kw.get("fine_grained", True)

        # feature embeddings
        self.width_emb   = nn.Embedding(self.max_span_width, 20)
        self.width_prior = nn.Embedding(self.max_span_width, 20)
        self.speaker_emb = nn.Embedding(2, 20)
        self.segment_emb = nn.Embedding(3, 20)
        self.genre_emb   = nn.Embedding(7, 20)
        self.dist_emb    = nn.Embedding(10, 20)
        # head attention
        H = self.bert.config.hidden_size
        self.head_attn = HeadAttention(H)
        # networks
        M, A = 2324, 7052
        ffn = kw.get("ff_hidden", 3000)
        self.mention_ffnn = nn.Sequential(
            nn.Linear(M, ffn), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(ffn, 1)
        )
        self.width_ffnn = nn.Sequential(
            nn.Linear(20, ffn), nn.ReLU(),
            nn.Linear(ffn, 1)
        )
        CE = M + 20 + 20 + 20
        self.coarse_proj1 = nn.Linear(CE*2, M)
        self.coarse_proj2 = nn.Linear(M, M)
        self.ant_ffnn     = nn.Sequential(
            nn.Linear(A, ffn), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(ffn, 1)
        )

    @classmethod
    def from_tf_checkpoint(cls, ckpt, cfg, device="cpu", **kw):
        bert = load_spanbert_backbone_from_tf(ckpt, cfg, device)
        m = cls(bert.config, **kw)
        m.bert = bert.to(device)
        m   = load_coref_head_from_tf(m, ckpt)
        return m.to(device).eval()

    def preprocess_text(self, text: str):
        def is_ws(c): return c in (" ", "\t", "\r", "\n", chr(0x202F))
        def is_pc(c): return c in (
            ".", ",", "", "\"", "'", "(", ")", "-", "/", "\\", "*", ";", ":")
        doc=[]; pw=pp=True
        for c in text:
            if is_pc(c): pp,doc=True,doc + [c]
            elif is_ws(c): pw=True
            else:
                if pw or pp: doc.append(c)
                else: doc[-1]+=c
                pw=pp=False
        toks=["[CLS]"]; sm=[0]; sent=0
        for w in doc:
            pieces = self.tokenizer.tokenize(w)
            toks += pieces; sm += [sent]*len(pieces)
            if w == ".": sent += 1
        toks += ["[SEP]"]; sm += [sm[-1]]
        return toks, sm

    def create_spans(self, toks):
        out=[]; L=len(toks); W=self.max_span_width
        for i in range(L):
            for j in range(i, min(i+W, L)):
                out.append([i,j])
        return out

    def _bucket_distance_1d(self, distances):
        """Bucket distances into 10 buckets"""
        buckets = torch.zeros_like(distances)
        buckets[distances == 0] = 0
        buckets[(distances >= 1) & (distances <= 1)] = 1
        buckets[(distances >= 2) & (distances <= 2)] = 2
        buckets[(distances >= 3) & (distances <= 3)] = 3
        buckets[(distances >= 4) & (distances <= 4)] = 4
        buckets[(distances >= 5) & (distances <= 7)] = 5
        buckets[(distances >= 8) & (distances <= 15)] = 6
        buckets[(distances >= 16) & (distances <= 31)] = 7
        buckets[(distances >= 32) & (distances <= 63)] = 8
        buckets[(distances >= 64)] = 9
        return buckets.long()

    def _embed_spans(self, h, spans, spk_ids, gen_ids, sent_map):
        B,L,Hd = h.size(); N=spans.size(1)
        s = spans[:,:,0].clamp(0,L-1)
        e = spans[:,:,1].clamp(0,L-1)
        start = h.gather(1, s.unsqueeze(-1).expand(-1,-1,Hd))
        end   = h.gather(1, e.unsqueeze(-1).expand(-1,-1,Hd))
        toks,mask = extract_span_tokens(h, spans, self.max_span_width)
        head = self.head_attn(toks, mask).view(B,N,Hd)
        widths = (e - s + 1).clamp(1,self.max_span_width) - 1
        wemb   = self.width_emb(widths)
        base = torch.cat([start, end, head, wemb], -1)
        spk = spk_ids.gather(1,s) if spk_ids is not None else torch.zeros(B,N,device=h.device,dtype=torch.long)
        seg = sent_map.gather(1,s)%3 if sent_map is not None else torch.zeros(B,N,device=h.device,dtype=torch.long)
        gen = gen_ids.unsqueeze(1).expand(-1,N) if gen_ids is not None else torch.zeros(B,N,device=h.device,dtype=torch.long)
        coarse = torch.cat([
            base,
            self.speaker_emb(spk),
            self.segment_emb(seg),
            self.genre_emb(gen)
        ], -1)
        return base, coarse, widths

    def forward(self, input_ids, attn_mask, spans, spk_ids=None, gen_ids=None, sent_map=None):
        B = input_ids.size(0)
        device = input_ids.device
        
        # 1) BERT encoding
        h = self.bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        # 2) Embed spans
        base, coarse, widths = self._embed_spans(h, spans, spk_ids, gen_ids, sent_map)

        # 3) Mention + width scoring
        ment_logits  = self.mention_ffnn(base).squeeze(-1)
        width_logits = self.width_ffnn(self.width_prior(widths)).squeeze(-1)
        mention_scores = F.softmax(ment_logits + width_logits, dim=1)

        # 4) Top-span pruning
        pruned = []
        for b in range(B):
            k = max(1, int((attn_mask[b].sum() - 2).item()* self.top_span_ratio))
            _, idx = mention_scores[b].sort(descending=True)
            used, keep = torch.zeros_like(attn_mask[b]), []
            for i in idx:
                s,e = spans[b,i]
                if used[s:e+1].any(): 
                    continue
                keep.append(i)
                used[s:e+1] = 1
                if len(keep) == k: 
                    break
                
            pruned.append(torch.tensor(keep, device=device))

        maxk = max(len(x) for x in pruned)
        pt = torch.stack([
            torch.cat([x, x.new_zeros(maxk - len(x))]) for x in pruned])
        spans_pruned = torch.gather(spans, 1,pt.unsqueeze(-1).expand(-1,-1,2))
        emb_pruned   = torch.gather(base, 1, pt.unsqueeze(-1).expand(-1,-1,base.size(-1)))
        coarse_pruned = torch.gather(coarse, 1, pt.unsqueeze(-1).expand(-1,-1,coarse.size(-1)))

        # 5) Pairwise & coarse_scores
        pairs = [(b,i,j)
                 for b in range(B)
                 for i in range(maxk)
                 for j in range(i)]
        if not pairs:
            return {
                "mention_scores":       mention_scores,
                "spans_pruned":         spans_pruned,
                "predicted_clusters":   []
                }
        bi,ii,jj = zip(*pairs)
        bi,ii,jj = map(lambda t: torch.tensor(t, device=device), (bi,ii,jj))
        ci = coarse_pruned[bi,ii]; cj = coarse_pruned[bi,jj]
        cp = torch.cat([ci, cj], dim=-1)
        h1 = self.coarse_proj1(cp); h2 = self.coarse_proj2(h1)
        coarse_scores = h2.sum(-1)  # [#pairs]

        # 6) Build k×k coarse matrix & top-c beam
        k = maxk
        coarse_mat = torch.full((k,k), float('-inf'), device=device)
        ptr = 0
        for i in range(k):
            for j in range(i):
                coarse_mat[i,j] = coarse_scores[ptr]
                ptr += 1

        c = min(self.max_top_ant, k)
        top_scores, top_inds = coarse_mat.topk(c, dim=1)  # both [k, c]

        # 6.5) FINE-GRAINED SCORING
        fine_scores = None
        if self.fine_grained and c > 0:
            fine_scores = torch.zeros(B, k, c, device=device)
            
            for b in range(B):
                valid_k = len(pruned[b])
                mention_embs = emb_pruned[b, :valid_k]  # [valid_k, 2324]
                
                for i in range(valid_k):
                    m_emb = mention_embs[i]  # [2324]
                    valid_c = min(i, c)
                    
                    if valid_c > 0:
                        ant_indices = top_inds[i, :valid_c]
                        ant_embs = mention_embs[ant_indices]  # [valid_c, 2324]
                        
                        # Distance features
                        m_pos = pt[b, i]
                        ant_pos = pt[b, ant_indices]
                        distances = m_pos - ant_pos
                        dist_buckets = self._bucket_distance_1d(distances)
                        dist_embs = self.dist_emb(dist_buckets)  # [valid_c, 20]
                        
                        # Similarity features
                        m_emb_exp = m_emb.unsqueeze(0).expand(valid_c, -1)
                        similarity = m_emb_exp * ant_embs  # [valid_c, 2324]
                        
                        # Zero placeholders
                        zero_20 = torch.zeros(valid_c, 20, device=device)
                        
                        # Build fine features
                        fine_features = torch.cat([
                            m_emb_exp,    # [valid_c, 2324]
                            ant_embs,     # [valid_c, 2324]
                            similarity,   # [valid_c, 2324]
                            dist_embs,    # [valid_c, 20]
                            zero_20,      # [valid_c, 20] speaker
                            zero_20,      # [valid_c, 20] genre
                            zero_20       # [valid_c, 20] segment
                        ], dim=1)        # [valid_c, 7052]
                        
                        # Apply ant_ffnn
                        f_scores = self.ant_ffnn(fine_features).squeeze(-1)
                        fine_scores[b, i, :valid_c] = f_scores

        # 7) Combined scoring & thresholded linking
        predicted_antecedents = []
        link_probs = []
        
        for b in range(B):
            b_antecedents = []
            b_probs = []
            valid_k = len(pruned[b]) if b < len(pruned) else 0
            
            for i in range(k):
                if i < valid_k:
                    coarse_k = top_scores[i]  # [c]
                    
                    if fine_scores is not None:
                        fine_k = fine_scores[b, i]  # [c]
                        combined = 0.5 * coarse_k + 0.5 * fine_k
                    else:
                        combined = coarse_k
                    
                    scores_with_dummy = torch.cat([
                        self.dummy_bias.view(1),
                        combined
                    ], dim=0)
                    
                    probs = F.softmax(scores_with_dummy / self.ant_temp, dim=0)
                    b_probs.append(probs)
                    
                    best_idx = torch.argmax(probs).item()
                    best_prob = probs[best_idx].item()
                    
                    if best_idx > 0 and best_prob > self.link_threshold:
                        ant_idx = top_inds[i, best_idx-1].item()
                        b_antecedents.append(ant_idx)
                    else:
                        b_antecedents.append(-1)
                else:
                    b_antecedents.append(-1)
            
            predicted_antecedents.append(b_antecedents)
            link_probs.append(b_probs)

        # 8) Build clusters
        mention_to_pred = {}
        predicted_clusters = []
        
        for b in range(B):
            if b < len(pruned):
                valid_k = len(pruned[b])
                for i in range(valid_k):
                    ant_idx = predicted_antecedents[b][i]
                    if ant_idx >= 0:
                        mention = tuple(spans_pruned[b,i].tolist())
                        antecedent = tuple(spans_pruned[b,ant_idx].tolist())
                        
                        if antecedent in mention_to_pred:
                            cid = mention_to_pred[antecedent]
                        else:
                            cid = len(predicted_clusters)
                            mention_to_pred[antecedent] = cid
                            predicted_clusters.append([antecedent])
                        
                        if mention not in mention_to_pred:
                            mention_to_pred[mention] = cid
                            predicted_clusters[cid].append(mention)

        return {
            "mention_scores":         mention_scores,
            "spans_pruned":           spans_pruned,
            "coarse_scores":          coarse_scores,
            "fine_scores":            fine_scores,
            "top_antecedents":        top_inds.unsqueeze(0),
            "top_antecedent_scores":  top_scores.unsqueeze(0),
            "predicted_antecedents":  predicted_antecedents,
            "predicted_clusters":     predicted_clusters,
            "link_probs":             link_probs
        }


# ---------------------------------------------------
# predict_file CLI
# ---------------------------------------------------
def predict_file(ckpt, bert_cfg, inp_path, out_path, device="cpu"):
    # 1) Load model from TF checkpoint
    CFG = {
        "ff_hidden":      3000,
        "max_span_width": 30,
        "top_span_ratio": 0.5,
        "max_top_ant":    50,
        "ant_temp":       0.5,
        "dummy_bias":     -1.0,
        "link_threshold": 0.5,
        "fine_grained":   True  # Enable fine-grained scoring
    }
    model = SpanBERTFullCoref.from_tf_checkpoint(ckpt, bert_cfg, device=device, **CFG)
    model.to(device).eval()

    # 2) Tokenizer for preprocessing & detokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    with open(inp_path) as fin, open(out_path, "w") as fout:
        for example_num, line in enumerate(fin):
            example = json.loads(line)
            text = example.get("text", "")

            # Preprocess
            toks, sent_map = model.preprocess_text(text)
            enc = model.tokenizer(
                toks,
                is_split_into_words=True,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)

            # Build spans [1, num_spans, 2]
            spans = torch.tensor(
                [model.create_spans(toks)],
                dtype=torch.long,
                device=device
            )
            # Build sentence map [1, seq_len]
            sm_t = torch.tensor([sent_map], dtype=torch.long, device=device)

            # 3) Run the model
            with torch.no_grad():
                outd = model(
                    input_ids,
                    attn_mask,
                    spans,
                    spk_ids=None,
                    gen_ids=None,
                    sent_map=sm_t
                )

            # DEBUG: print top-10 mention candidates
            scores, idxs = torch.topk(outd["mention_scores"][0],
                                      k=min(10, outd["mention_scores"].size(1)))
            raw_spans = spans[0]
            tokens    = tokenizer.convert_ids_to_tokens(input_ids[0])
            print(f"\nExample {example_num} top-10 mentions:")
            for score, idx in zip(scores.tolist(), idxs.tolist()):
                s,e = raw_spans[idx].tolist()
                span_pieces = tokens[s:e+1]
                detok = ""
                for piece in span_pieces:
                    if piece.startswith("##"):
                        detok += piece[2:]
                    else:
                        detok += (" " + piece) if detok else piece
                print(f"  {score:.3f}: '{detok}'  (span idxs {s},{e})")

            # 4) Assemble JSON output
            pr = outd["spans_pruned"][0].tolist()
            example["top_spans"]          = [tuple(span) for span in pr]
            example["predicted_clusters"] = outd.get("predicted_clusters", [])
            example["head_scores"]        = []

            fout.write(json.dumps(example) + "\n")

            if example_num % 100 == 0:
                print(f"Processed {example_num + 1} examples.")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py ckpt bert_cfg inp_path out_path [device]")
        sys.exit(1)
    
    _, ckpt, bert_cfg, inp_path, out_path = sys.argv[:5]
    device = sys.argv[5] if len(sys.argv) > 5 else "cpu"
    predict_file(ckpt, bert_cfg, inp_path, out_path, device)