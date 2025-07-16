# =============================================================================
# keyphrase_analysis.py - Keyphrase-specific analysis
# =============================================================================

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from spanbert_kp_extractor import BertKpeExtractor
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
from difflib  import SequenceMatcher
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
    nlp
)

class KeyphraseAnalysisInterface:
    def __init__(self):
        self.results = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        self.setup_interface()
        
    
    def setup_models(self):
        """Initialize keyphrase-specific models"""
        print("Loading keyphrase extraction models...")
        self.kpe_extractor = BertKpeExtractor(
            checkpoint_path=BertKpeExtractor.get_default_checkpoint_path(),
            bert_kpe_repo_path=BertKpeExtractor.get_default_repo_path()
        )
        self.kpe_extractor.device = self.device
        print("✅ Keyphrase models loaded")
    
    load_models = setup_models
    
    def setup_interface(self):
        """Create keyphrase analysis interface"""
        
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
        
        self.top_k_phrases = widgets.IntSlider(
            value=5, min=1, max=10, step=1,
            description='Top K Phrases:'
        )
        
        self.results_output = widgets.Output(
        layout=widgets.Layout(overflow='auto', height='400px')
        )
    
        self.selector_container = widgets.VBox(
        [], layout=widgets.Layout(overflow='auto', height='400px')
        )
        
        # Run button
        self.run_btn = widgets.Button(
            description='🚀 Run Keyphrase Analysis',
            button_style='primary',
            layout=widgets.Layout(width='250px')
        )
        self.run_btn.on_click(self.run_analysis)
        
                
        # Main interface
        self.interface = widgets.VBox([
            widgets.HTML("<h3>🔑 Keyphrase Analysis Configuration</h3>"),
            self.file_input,
            self.max_sentences,
            self.top_k_phrases,
            self.run_btn,
            self.results_output,
            self.selector_container
        ])
        
    def show(self):
        """Display this interface in a notebook."""
        from IPython.display import display
        display(self.interface)

        
    
    def extract_keyphrases_batch(self, sentences, top_k=5, threshold=0.1):
        """Extract keyphrases from sentences using BERT-KPE"""
        print(f"🔑 Extracting up to {top_k} keyphrases from {len(sentences)} sentences…")
        
        raw_kp_lists = self.kpe_extractor.batch_extract_keyphrases(
            texts=sentences,
            top_k=top_k,
            threshold=threshold
        )
        
        results = []
        for i, (sentence, kp_list) in enumerate(zip(sentences, raw_kp_lists)):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i+1}/{len(sentences)}")
            
            formatted_kps = []
            for rank, (phrase, score) in enumerate(kp_list, start=1):
                formatted_kps.append({
                    'rank': rank,
                    'phrase': phrase,
                    'score': score
                })
            
            if formatted_kps:
                results.append({
                    'sentence': sentence,
                    'keyphrases': formatted_kps
                })
        
        print(f"✅ Successfully extracted keyphrases from {len(results)} sentences")
        return results
    

    
    def analyze_keyphrases_with_coreference(
        self,
        sentence: str,
        full_text: str,
        keyphrases: List[str],
        all_chains: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
        """
        For each keyword/phrase in `keyphrases`, locate it in the sentence/full_text,
        optionally expand via dependency parsing, normalize into candidate spans,
        then map to your precomputed coreference chains (`all_chains`) using
        overlap + head-noun fallback exactly as before.
        """
        # 1) locate or fuzzy-match the sentence in full_text
        doc = nlp(full_text)
        best_sent, best_score = max(
            ((s, SequenceMatcher(None, s.text, sentence).ratio()) for s in doc.sents),
            key=lambda x: x[1]
            )
        if best_score < 0.5:
            sent_start = full_text.find(sentence)
            if sent_start == -1:
                doc_sents = sent_tokenize(full_text)
                best = max(doc_sents, key=lambda s: SequenceMatcher(None, s, sentence).ratio())
                score = SequenceMatcher(None, best, sentence).ratio()
                if score < 0.5:
                    return []
                sentence = best
                sent_start = full_text.find(best)
        else:
            sent_start = best_sent.start_char
            sentence = best_sent.text

        enhanced = []
        for phrase in keyphrases:
            local_idx = sentence.lower().find(phrase.lower())
            if local_idx < 0:
                continue
            abs_start = sent_start + local_idx
            abs_end = abs_start + len(phrase)

            expanded, (ls, le) = expand_to_full_phrase(sentence, local_idx, local_idx + len(phrase))
            g_start = sent_start + ls
            g_end = sent_start + le

            candidates = normalize_span_for_chaining(sentence, ls, le)

            # 5) overlap-based chain mapping
            matched_chain = None
            matched_span = None
            for txt, s_off, e_off in candidates:
                a = sent_start + s_off
                b = sent_start + e_off
                for chain in all_chains:
                    mentions = []
                    if 'mentions' in chain and chain['mentions']:
                        if isinstance(chain['mentions'][0], dict):
                            mentions = [(m['start_char'], m['end_char'], m['text']) for m in chain['mentions']]
                        else:
                            mentions = [(m[0], m[1], m[2]) for m in chain['mentions']]
                    if mentions and any(a < m1 and b > m0 for (m0, m1, _) in mentions):
                        matched_chain = chain
                        matched_span = (txt, a, b)
                    break
                if matched_chain:
                    break

            # 6) HEAD-NOUN fallback
            if not matched_chain:
                span_doc = nlp(expanded)
                roots = [tok for tok in span_doc if tok.dep_ == 'ROOT']
                head = roots[0] if roots else (span_doc[0] if span_doc else None)
                if head and head.pos_ != 'NOUN':
                    head = next((t for t in span_doc if t.pos_ == 'NOUN'), head)
                if head:
                    htext = head.text.lower()
                    for chain in all_chains:
                        rep = chain['representative'].lower()
                        if 'mentions' in chain:
                            if isinstance(chain['mentions'][0], dict):
                                mention_texts = [m['text'].lower() for m in chain['mentions']]
                            else:
                                mention_texts = [m[2].lower() for m in chain['mentions']]
                        else:
                            mention_texts = []
                        if rep == htext or htext in mention_texts:
                            matched_chain = chain
                            idx = sentence.lower().find(htext)
                            if idx >= 0:
                                a = sent_start + idx
                                matched_span = (htext, a, a + len(htext))
                            break

        coref_info = {"chain_found": bool(matched_chain)}
        if matched_chain:
            related_mentions = []
            for m in matched_chain['mentions']:
                if isinstance(m, dict):
                    m_start, m_end, m_text = m['start_char'], m['end_char'], m['text']
                else:
                    m_start, m_end, m_text = m[0], m[1], m[2]
                if matched_span and not (m_start == matched_span[1] and m_end == matched_span[2]):
                    related_mentions.append({"text": m_text, "coords": (m_start, m_end)})
            coref_info.update({
                "chain_id": matched_chain["chain_id"],
                "representative": matched_chain["representative"],
                "related_mentions": related_mentions
            })

        enhanced.append({
            "phrase": phrase,
            "expanded_phrase": expanded,
            "coords": (g_start, g_end),
            "coreference_analysis": coref_info
        })
    
        return enhanced

    def run_keyphrase_pipeline(
        self,
        pdf_path,
        max_sentences=50,
        top_k=5,
        threshold=0.1,
        agree_thresh=0.1,
        disagree_thresh=0.2,
        min_conf=0.5
    ):
        """Main keyphrase pipeline"""
        print("🚀 Starting Keyphrase Analysis Pipeline")
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
        
        # Extract keyphrases
        print(f"\n🔑 Extracting keyphrases from {len(sents)} sentences...")
        kp_results = self.extract_keyphrases_batch(sents, top_k=top_k, threshold=threshold)
        
        # Process each sentence
        results = []
        for s in sents:
            
            b_lab, b_conf = classify_sentence_bert(s)
            si_lab, si_conf = classify_sentence_similarity(s)
            # pass your tuned thresholds into the consensus function
            cons = determine_dual_consensus(
                b_lab, b_conf,
                si_lab, si_conf,
                agree_thresh, disagree_thresh, min_conf
            )
            # Find keyphrases for this sentence
            sentence_kps = []
            for kp_entry in kp_results:
                if kp_entry['sentence'] == s:
                    sentence_kps = [kp['phrase'] for kp in kp_entry['keyphrases']]
                    break
            
            # Enhanced keyphrase-coreference analysis
            keyphrase_analysis = []
            if sentence_kps:
                keyphrase_analysis = self.analyze_keyphrases_with_coreference(
                    sentence=s,
                    full_text=clean,
                    keyphrases=sentence_kps,
                    all_chains=all_chains
                )
            
            results.append({
                "sentence": s,
                "primary_result": {"label": b_lab, "confidence": b_conf},
                "secondary_result": {"label": si_lab, "confidence": si_conf},
                "consensus": cons,
                "keyphrase_analysis": keyphrase_analysis
            })
        
        
        output = {
            "results":      results,
            "full_text":    clean,
            "coref_chains": all_chains,
            "keyphrase_summary": {
                "total_sentences_with_keyphrases": sum(1 for r in results if r["keyphrase_analysis"]),
                "total_keyphrases": sum(len(entry["keyphrases"]) for entry in kp_results)
                }
            }   
    
    
        sentence_analyses = [
            {
                "sentence_id":       idx,
                "sentence_text":     r["sentence"],
                "keyphrase_analysis": r["keyphrase_analysis"]
                }
            for idx, r in enumerate(results)
        ]
        
        from helper import list_and_filter_coref_clusters_from_kpe, build_cluster_graphs
        
        clusters = list_and_filter_coref_clusters_from_kpe(sentence_analyses)
        cluster_graphs = build_cluster_graphs({c["chain_id"]: c for c in clusters})
    
        output["clusters"] = clusters
        output["cluster_graphs"] = cluster_graphs
    
        return output
        
        
    def run_analysis(self, b):
        """Run the keyphrase analysis"""
        with self.results_output:
            clear_output(wait=True)
            
            if not self.file_input.value:
                print("❌ Please enter a PDF file path")
                return
            
            try:
                # Run pipeline
                self.results = self.run_keyphrase_pipeline(
                    self.file_input.value,
                    max_sentences=self.max_sentences.value,
                    top_k=self.top_k_phrases.value
                )
                
                
            except Exception as e:
                print(f"❌ Error: {e}")
        if self.results:
            dd, out = create_selector(self.results)
            self.selector_container.children = [dd, out]  
              
    def get_interface(self):
        """Return the main interface widget"""
        return self.interface


run_pipeline = KeyphraseAnalysisInterface().run_keyphrase_pipeline         # add at line ~361

# Expose the coreference method as a module‐level function for the launcher
from keyphrase_analysis import KeyphraseAnalysisInterface

analyze_keyphrases_with_coreference = (
    KeyphraseAnalysisInterface()
    .analyze_keyphrases_with_coreference
)
