# =============================================================================
# span_analysis.py - IG Span-specific analysis  
# =============================================================================

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from difflib  import SequenceMatcher
from typing import List, Dict, Any
import spacy 
import coreferee


from helper import (
    extract_text_from_pdf_robust,
    preprocess_pdf_text,
    extract_and_filter_sentences,
    classify_sentence_bert,
    classify_sentence_similarity,
    determine_dual_consensus,
    analyze_full_text_coreferences,
    print_explanation_summary,
    create_selector,
    expand_to_full_phrase, 
    normalize_span_for_chaining,
    bert_model, 
    bert_tokenizer,
    nlp

)



class SpanAnalysisInterface:
    def __init__(self):
        self.results = None
        self.setup_interface()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = bert_model
        self.bert_model = bert_model.to(self.device)
        self.bert_tokenizer = bert_tokenizer 
        
    
    def setup_interface(self):
        """Create span analysis interface"""
        
        # File input
        self.file_input = widgets.Text(
            placeholder='Enter PDF file path...',
            description='PDF Path:',
            layout=widgets.Layout(width='500px')
        )
        
        # Parameters
        self.max_sentences = widgets.IntSlider(
            value=30, min=10, max=100, step=5,
            description='Max Sentences:'
        )
        
        self.max_span_len = widgets.IntSlider(
            value=4, min=1, max=8, step=1,
            description='Max Span Length:'
        )
        
        self.top_k_spans = widgets.IntSlider(
            value=5, min=1, max=10, step=1,
            description='Top K Spans:'
        )
        
        self.results_output = widgets.Output(
        layout=widgets.Layout(overflow='auto', height='400px')
        )
        
        self.selector_container = widgets.VBox(
        [], layout=widgets.Layout(overflow='auto', height='400px')
        )
        
        # Run button
        self.run_btn = widgets.Button(
            description='🚀 Run Span Analysis',
            button_style='success',
            layout=widgets.Layout(width='250px')
        )
        self.run_btn.on_click(self.run_analysis)
        
        # Output area
        self.output = widgets.Output()
        
        # Results selector (initially empty)
        self.results_selector = widgets.VBox([])
        
        # Main interface
        self.interface = widgets.VBox([
            widgets.HTML("<h3>🎯 IG Span Analysis Configuration</h3>"),
            self.file_input,
            self.max_sentences,
            self.max_span_len,
            self.top_k_spans,
            self.run_btn,
            self.results_output,
            self.selector_container
        ])
        
        
    def show(self):
        """Display this interface in a notebook."""
        from IPython.display import display
        display(self.interface)

        
    def get_token_importance_ig(self, text, target_class):
        enc = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids, attn_mask = enc.input_ids, enc.attention_mask
        def forward_fn(emb, mask):
            return bert_model(inputs_embeds=emb, attention_mask=mask).logits
        
        embed = bert_model.bert.embeddings.word_embeddings(input_ids)
        ig    = IntegratedGradients(forward_fn)
        atts, _ = ig.attribute(
            inputs=embed.to(self.device),
            baselines=torch.zeros_like(embed).to(self.device),
            target=target_class,
            additional_forward_args=(attn_mask,),
            return_convergence_delta=True
            )
        tokens = bert_tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        scores = atts.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        return tokens, scores

    
    def compute_span_importances(self, text, target_class, max_span_len=4):
        enc = bert_tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        input_ids = enc.input_ids.clone().to(self.device)
        attn_mask = enc.attention_mask.to(self.device)
        offsets = enc.offset_mapping[0].tolist()
        tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0])
        with torch.no_grad():
            orig_logit = bert_model(input_ids=input_ids, attention_mask=attn_mask).logits[0, target_class].item()
            span_scores = []
            n = len(tokens)
            for i in range(1, n-1):
                for j in range(i+1, min(i+1+max_span_len, n-1)):
                    s_char = offsets[i][0]
                    e_char = offsets[j-1][1]
                    if s_char < e_char:
                        masked = input_ids.clone()
                        m_id = bert_tokenizer.mask_token_id
                        for k in range(i, j):
                            masked[0, k] = m_id
                        with torch.no_grad():
                            new_logit = bert_model(input_ids=masked, attention_mask=attn_mask).logits[0, target_class].item()
                            delta = orig_logit - new_logit
                            span_text = " ".join(tokens[i:j])
                            span_scores.append({'text': span_text,
                                     'importance': delta,
                                     'start_char': s_char,
                                     'end_char': e_char,
                                     'start_tok': i,
                                     'end_tok': j})
        return span_scores, tokens, offsets

    
    def compute_span_importances_ig(self, text: str, target_class: int):
        enc = bert_tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=512
            ).to(device)
        offsets = enc.offset_mapping[0].tolist()
        tokens  = bert_tokenizer.convert_ids_to_tokens(enc.input_ids.squeeze(0))
        _, scores = get_token_importance_ig(text, target_class)
        span_scores = [{
            "text": tokens[i],
            "importance": float(scores[i]),
            "start_char": offsets[i][0],
            "end_char":   offsets[i][1],
            "start_tok":  i, "end_tok": i+1
            } for i in range(len(tokens))]
        return span_scores, tokens, offsets
    
    
    
    def run_span_pipeline(self, pdf_path, max_sentences=50, max_span_len=4):
        """Main span analysis pipeline"""
        print("🚀 Starting IG Span Analysis Pipeline")
        print("=" * 70)
        
        # Extract and preprocess text
        raw = extract_text_from_pdf_robust(pdf_path)
        clean = preprocess_pdf_text(raw)
        
        # Extract coreference chains
        print("\n🔍 Analyzing coreference chains...")
        chains_result = analyze_full_text_coreferences(clean)
        all_chains = chains_result["chains"]
        
        # Split sentences
        sents = extract_and_filter_sentences(clean)
        if len(sents) > max_sentences:
            print(f"⚠️  Limiting to first {max_sentences} sentences")
            sents = sents[:max_sentences]
        
        results = []
        for s in sents:
            # Classification
            b_lab, b_conf = classify_sentence_bert(s)
            si_lab, si_conf = classify_sentence_similarity(s)
            cons = determine_dual_consensus(b_lab, b_conf, si_lab, si_conf)
            
            # Extract important spans using IG
            span_scores, tokens, offsets = self.compute_span_importances(s, b_lab, max_span_len)
            top_spans = sorted(span_scores, key=lambda x: -x['importance'])[:5]
            
            # Enhanced span-coreference analysis
            span_analysis = []
            if top_spans:
                span_analysis = self.analyze_spans_with_coreference(
                    sentence=s,
                    full_text=clean,
                    spans=top_spans,
                    all_chains=all_chains
                )
            
            results.append({
                "sentence": s,
                "primary_result": {"label": b_lab, "confidence": b_conf},
                "secondary_result": {"label": si_lab, "confidence": si_conf},
                "consensus": cons,
                "span_analysis": span_analysis
            })
        
        
        output = {
            "results":      results,
            "full_text":    clean,
            "coref_chains": all_chains,
            "span_summary": {
                "total_sentences_with_spans": sum(1 for r in results if r["span_analysis"]),
                "total_spans": sum(len(r["span_analysis"]) for r in results)
            }
        }
 
        
        sentence_analyses = [
            {
                "sentence_id":     idx,
                "sentence_text":   r["sentence"],
                "span_analysis":   r["span_analysis"]
                }
            for idx, r in enumerate(results)
            ]  
        
        from helper import list_and_filter_coref_clusters_from_kpe, build_cluster_graphs
        
        clusters       = list_and_filter_coref_clusters_from_kpe(sentence_analyses)
        cluster_graphs = build_cluster_graphs({c["chain_id"]: c for c in clusters})
 
        output["clusters"]       = clusters
        output["cluster_graphs"] = cluster_graphs
 
        return output
        
        
    
    def analyze_spans_with_coreference(
        self,  
        sentence: str,
        full_text: str,
        spans: List[Dict],
        all_chains: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
        """
        For each important span in `spans`, locate it in the sentence/full_text,
        optionally expand via dependency parsing, normalize into candidate spans,
        then map to your precomputed coreference chains (`all_chains`) using
        overlap + head-noun fallback exactly as in keyphrase version.
        """
        # 1) locate or fuzzy-match the sentence in full_text
        sent_start = full_text.find(sentence)
        if sent_start == -1:
            doc_sents = sent_tokenize(full_text)
            best = max(doc_sents, key=lambda s: SequenceMatcher(None, s, sentence).ratio())
            score = SequenceMatcher(None, best, sentence).ratio()
            if score < 0.5:
                # no good match → nothing to do
                return []
            sentence = best
            sent_start = full_text.find(best)

        enhanced = []
        for span in spans:
            # 2) get span coordinates within sentence
            local_start = span['start_char']
            local_end = span['end_char']
            span_text = span['text']

            # 3) expand to a full dependency-based phrase if it was truncated
            expanded, (ls, le) = expand_to_full_phrase(sentence, local_start, local_end)
            g_start = sent_start + ls
            g_end   = sent_start + le

            # 4) normalization → candidate spans
            candidates = normalize_span_for_chaining(sentence, ls, le)

            # 5) overlap-based chain mapping
            matched_chain = None
            matched_span  = None
            for txt, s_off, e_off in candidates:
                a = sent_start + s_off
                b = sent_start + e_off
                for chain in all_chains:
                    # Convert chain mentions to the expected format
                    if 'mentions' in chain and len(chain['mentions']) > 0:
                        # Handle different mention formats
                        if isinstance(chain['mentions'][0], dict):
                            # Format: [{'start_char': x, 'end_char': y, 'text': z}, ...]
                            mentions = [(m['start_char'], m['end_char'], m['text']) for m in chain['mentions']]
                        else:
                            # Format: [(start, end, text), ...]
                            mentions = chain['mentions']

                    # Check for overlap
                        if any(not (b <= m0 or a >= m1) for (m0, m1, _) in mentions):
                            matched_chain = chain
                            matched_span  = (txt, a, b)
                            break
            if matched_chain:
                break

        # 6) HEAD-NOUN fallback
        if not matched_chain:
            span_doc = nlp(expanded)
            roots = [tok for tok in span_doc if tok.dep_ == 'ROOT']
            head = roots[0] if roots else (span_doc[0] if span_doc else None)
            if head and head.pos_ != "NOUN":
                head = next((t for t in span_doc if t.pos_ == "NOUN"), head)
            if head:
                htext = head.text.lower()
                for chain in all_chains:
                    rep = chain["representative"].lower()
                    # Handle different mention formats for head-noun matching
                    if 'mentions' in chain:
                        if isinstance(chain['mentions'][0], dict):
                            mention_texts = [m['text'].lower() for m in chain['mentions']]
                        else:
                            mention_texts = [m[2].lower() for m in chain['mentions']]
                    else:
                        mention_texts = []

                    if rep == htext or any(mtext == htext for mtext in mention_texts):
                        matched_chain = chain
                        idx = sentence.lower().find(htext)
                        if idx >= 0:
                            a = sent_start + idx
                            matched_span = (htext, a, a + len(htext))
                        break

        # 7) build coref_info
        coref_info = {"chain_found": bool(matched_chain)}
        if matched_chain:
            # Handle different mention formats for related mentions
            related_mentions = []
            if 'mentions' in matched_chain:
                for m in matched_chain['mentions']:
                    if isinstance(m, dict):
                        m_start, m_end, m_text = m['start_char'], m['end_char'], m['text']
                    else:
                        m_start, m_end, m_text = m[0], m[1], m[2]

                    # Skip if this is the matched span itself
                    if matched_span and not (m_start == matched_span[1] and m_end == matched_span[2]):
                        related_mentions.append({"text": m_text, "coords": (m_start, m_end)})

            coref_info.update({
                "chain_id":       matched_chain["chain_id"],
                "representative": matched_chain["representative"],
                "related_mentions": related_mentions
            })

        # 8) collect result for this span
        enhanced.append({
            "span":              span,
            "expanded_phrase":   {"text": expanded, "coords": (g_start, g_end)},
            "coreference_analysis": coref_info
        })

        return enhanced
    
    def run_analysis(self, b):
        """Run the span analysis"""
        with self.results_output:
            clear_output(wait=True)
            
            if not self.file_input.value:
                print("❌ Please enter a PDF file path")
                return
            
            try:
                # Run pipeline
                self.results = self.run_span_pipeline(
                    self.file_input.value,
                    max_sentences=self.max_sentences.value,
                    max_span_len=self.max_span_len.value
                )
                
                # Show summary
                print_explanation_summary(self.results)
                
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if self.results:
            dd, out = create_selector(self.results)
            self.selector_container.children = [dd, out]                
            
    def get_interface(self):
        """Return the main interface widget"""
        return self.interface

run_pipeline = SpanAnalysisInterface().run_span_pipeline

from span_analysis import SpanAnalysisInterface
span_compute_span_importances     = SpanAnalysisInterface().compute_span_importances
span_analyze_spans_with_coreference = SpanAnalysisInterface().analyze_spans_with_coreference
