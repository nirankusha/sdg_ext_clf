import argparse
import json
from dataclasses import dataclass
import torch
from spanbert_coref_extactor9 import SpanBERTFullCoref
from transformers import AutoTokenizer

@dataclass
class CFG:
    ckpt: str
    bert_cfg: str
    ff_hidden: int = 3000
    max_span_width: int = 30
    top_span_ratio: float = 0.4
    max_top_ant: int = 50
    ant_temp: float = 0.5
    dummy_bias: float = -1.0
    link_threshold: float = 0.5
    fine_grained: bool = True
    device: str = 'cuda'


def predict_file(cfg: CFG, inp_path: str, out_path: str):
    # Load and prepare model
    model = SpanBERTFullCoref.from_tf_checkpoint(
        cfg.ckpt,
        cfg.bert_cfg,
        device=cfg.device,
        ff_hidden=cfg.ff_hidden,
        max_span_width=cfg.max_span_width,
        top_span_ratio=cfg.top_span_ratio,
        max_top_ant=cfg.max_top_ant,
        ant_temp=cfg.ant_temp,
        dummy_bias=cfg.dummy_bias,
        link_threshold=cfg.link_threshold,
        fine_grained=cfg.fine_grained
    ).to(cfg.device).eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    with open(inp_path) as fin, open(out_path, 'w') as fout:
        for line in fin:
            example = json.loads(line)
            text = example.get("text", "")

            # Tokenize and map sentences
            toks, sent_map = model.preprocess_text(text)
            enc = tokenizer(
                toks,
                is_split_into_words=True,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(cfg.device)
            attn_mask = enc["attention_mask"].to(cfg.device)

            spans = torch.tensor([
                model.create_spans(toks)
            ], dtype=torch.long, device=cfg.device)
            sent_map_t = torch.tensor([
                sent_map
            ], dtype=torch.long, device=cfg.device)

            # Forward pass
            with torch.no_grad():
                outd = model(
                    input_ids,
                    attn_mask,
                    spans,
                    spk_ids=None,
                    gen_ids=None,
                    sent_map=sent_map_t
                )

                        # Serialize all keys from forward output recursively
            def serialize_obj(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().tolist()
                elif isinstance(obj, dict):
                    return {k: serialize_obj(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    cls = list if isinstance(obj, list) else tuple
                    return cls(serialize_obj(v) for v in obj)
                else:
                    return obj

            serialized = serialize_obj(outd)

            # Update example with serialized model outputs
            example.update(serialized)
            example.update(serialized)

            fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coref inference dumping all forward-output keys"
    )
    parser.add_argument("--ckpt", required=True,
                        help="TF checkpoint prefix (no extensions)")
    parser.add_argument("--bert_cfg", required=True,
                        help="BERT config JSON path")
    parser.add_argument("--input_file", required=True,
                        help="Input JSONL with 'text' field")
    parser.add_argument("--output_file", required=True,
                        help="Output JSONL path for predictions")
    parser.add_argument("--ff_hidden", type=int, default=3000,
                        help="FFNN hidden size")
    parser.add_argument("--max_span_width", type=int, default=30,
                        help="Max span width to consider")
    parser.add_argument("--top_span_ratio", type=float, default=0.4,
                        help="Ratio of spans to prune")
    parser.add_argument("--max_top_ant", type=int, default=50,
                        help="Max antecedents per span")
    parser.add_argument("--ant_temp", type=float, default=0.5,
                        help="Antecedent softmax temperature")
    parser.add_argument("--dummy_bias", type=float, default=-1.0,
                        help="Bias for dummy antecedent scores")
    parser.add_argument("--link_threshold", type=float, default=0.5,
                        help="Threshold for linking spans")
    parser.add_argument("--fine_grained", action="store_true",
                        help="Use fine-grained antecedent scoring")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device (cpu or cuda)")
    args = parser.parse_args()

    cfg = CFG(
        ckpt=args.ckpt,
        bert_cfg=args.bert_cfg,
        ff_hidden=args.ff_hidden,
        max_span_width=args.max_span_width,
        top_span_ratio=args.top_span_ratio,
        max_top_ant=args.max_top_ant,
        ant_temp=args.ant_temp,
        dummy_bias=args.dummy_bias,
        link_threshold=args.link_threshold,
        fine_grained=args.fine_grained,
        device=args.device
    )
    predict_file(cfg, args.input_file, args.output_file)
