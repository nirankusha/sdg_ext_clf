# =============================================================================
# main_tabbed_interface.py - Tabbed interface for SDG Analysis System
# =============================================================================
 
import ipywidgets as widgets
from IPython.display import display, clear_output
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any
from span_analysis import SpanAnalysisInterface
from keyphrase_analysis import KeyphraseAnalysisInterface
 
 
class TabbedSDGInterface:
     def __init__(self):
         self.setup_interface()
         
         
        
     def setup_interface(self):
         """Create the main tabbed interface"""
         
         # Create tabs
         tab = widgets.Tab()
         
         # Create containers for each tab
         keyphrase_container = widgets.VBox()
         span_container = widgets.VBox()
         comparison_container = widgets.VBox()
         
         # Set up tab children
         tab.children = [keyphrase_container, span_container, comparison_container]
         
         # Set tab titles
         tab.set_title(0, '🔑 Keyphrase Analysis')
         tab.set_title(1, '🎯 IG Span Analysis') 
         tab.set_title(2, '📊 Compare Results')
         
         # Initialize each tab
         self.initialize_keyphrase_tab(keyphrase_container)
         self.initialize_span_tab(span_container)
         self.initialize_comparison_tab(comparison_container)
         
         # Main interface
         self.main_interface = widgets.VBox([
             widgets.HTML("<h2>🎯 SDG Classification & Explainability System</h2>"),
             widgets.HTML("<p>Select a tab to choose your analysis method:</p>"),
             tab
         ])
         
         display(self.main_interface)
     
     def initialize_keyphrase_tab(self, container):
         """Initialize the keyphrase analysis tab"""
         
         # Loading message
         loading_output = widgets.Output()
         container.children = [loading_output]
         
         with loading_output:
             print("🔑 Initializing Keyphrase Analysis Module...")
         
         # Create keyphrase interface
         try:
             keyphrase_interface = KeyphraseTabInterface()
             container.children = [keyphrase_interface.get_interface()]
         except Exception as e:
             with loading_output:
                 clear_output()
                 print(f"❌ Error loading keyphrase analysis: {e}")
                 print("Make sure keyphrase_analysis.py and helper.py are available")
     
     def initialize_span_tab(self, container):
         """Initialize the span analysis tab"""
         
         # Loading message
         loading_output = widgets.Output()
         container.children = [loading_output]
         
         with loading_output:
             print("🎯 Initializing IG Span Analysis Module...")
         
         # Create span interface
         try:
             span_interface = SpanTabInterface()
             container.children = [span_interface.get_interface()]
         except Exception as e:
             with loading_output:
                 clear_output()
                 print(f"❌ Error loading span analysis: {e}")
                 print("Make sure span_analysis.py and helper.py are available")
     
     def initialize_comparison_tab(self, container):
         """Initialize the comparison tab"""
         comparison_interface = ComparisonTabInterface()
         container.children = [comparison_interface.get_interface()]
 
# =============================================================================
# Keyphrase Tab Interface
# =============================================================================
 
class KeyphraseTabInterface:
    def __init__(self):
        self.results = None
        self.kpe_extractor = None
        self.setup_interface()
        self.load_models()
        self.kp_iface = KeyphraseAnalysisInterface()
        self.analyze_keyphrases_with_coreference = (
           self.kp_iface.analyze_keyphrases_with_coreference
           )
     
    def load_models(self):
        """Load keyphrase-specific models"""
        try:
            from spanbert_kp_extractor import BertKpeExtractor
             
            with self.loading_output:
                print("Loading BERT-KPE model...")
                self.kpe_extractor = BertKpeExtractor(
                    checkpoint_path=BertKpeExtractor.get_default_checkpoint_path(),
                    bert_kpe_repo_path=BertKpeExtractor.get_default_repo_path()
                )
                clear_output()
                print("✅ Keyphrase models loaded successfully!")
                 
        except Exception as e:
            with self.loading_output:
                clear_output()
                print(f"❌ Error loading keyphrase models: {e}")
     
    def setup_interface(self):
        """Create the keyphrase analysis interface"""
         
        # Header
        header = widgets.HTML("""
        <div style='background:#e8f4fd; padding:15px; border-radius:8px; margin-bottom:15px;'>
            <h3 style='margin:0; color:#1f77b4;'>🔑 Keyphrase-based SDG Analysis</h3>
            <p style='margin:5px 0 0 0; color:#666;'>
                Extract keyphrases using BERT-KPE and analyze with coreference resolution.
            </p>
        </div>
        """)
         
        # File input section
        file_section = widgets.VBox([
            widgets.HTML("<h4>📁 Input Configuration</h4>"),
            widgets.HBox([
                widgets.Text(
                    placeholder='Enter PDF file path...',
                    description='PDF Path:',
                    layout=widgets.Layout(width='400px'),
                    style={'description_width': '80px'}
                ),
                widgets.Button(
                    description='📂 Browse',
                    button_style='info',
                    layout=widgets.Layout(width='80px')
                )
            ])
        ])
         
        self.file_input = file_section.children[1].children[0]
        browse_btn = file_section.children[1].children[1]
        browse_btn.on_click(self.browse_file)
         
        # Parameters section
        params_section = widgets.VBox([
            widgets.HTML("<h4>⚙️ Analysis Parameters</h4>"),
            widgets.HBox([
                widgets.IntSlider(
                    value=30, min=10, max=100, step=5,
                    description='Max Sentences:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                ),
                widgets.IntSlider(
                    value=5, min=1, max=10, step=1,
                    description='Top K Phrases:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                )
            ]),
            widgets.HBox([
                widgets.FloatSlider(
                    value=0.1, min=0.05, max=0.5, step=0.05,
                    description='KP Threshold:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                ),
                widgets.FloatSlider(
                    value=0.5, min=0.1, max=0.9, step=0.1,
                    description='Min Confidence:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                )
            ])
        ])
         
        self.max_sentences = params_section.children[1].children[0]
        self.top_k_phrases = params_section.children[1].children[1]
        self.kp_threshold = params_section.children[2].children[0]
        self.min_confidence = params_section.children[2].children[1]
         
        # Run section
        run_section = widgets.VBox([
            widgets.HTML("<h4>🚀 Execute Analysis</h4>"),
            widgets.HBox([
                widgets.Button(
                    description='🚀 Run Keyphrase Analysis',
                    button_style='primary',
                    layout=widgets.Layout(width='250px', height='40px')
                ),
                widgets.Button(
                    description='🔄 Reset',
                    button_style='warning',
                    layout=widgets.Layout(width='100px', height='40px')
                ),
                widgets.Button(
                    description='💾 Export Results',
                    button_style='success',
                    layout=widgets.Layout(width='150px', height='40px'),
                    disabled=True
                )
            ])
        ])
         
        self.run_btn = run_section.children[1].children[0]
        self.reset_btn = run_section.children[1].children[1]
        self.export_btn = run_section.children[1].children[2]
         
        self.run_btn.on_click(self.run_analysis)
        self.reset_btn.on_click(self.reset_interface)
        self.export_btn.on_click(self.export_results)
         
        # Output areas
        self.loading_output = widgets.Output(layout=widgets.Layout(height='100px'))
        self.results_output = widgets.Output()
        self.selector_container = widgets.VBox([])
         
        # Progress bar
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
         
        # Main container
        self.interface = widgets.VBox([
            header,
            file_section,
            params_section,
            run_section,
            self.progress_bar,
            self.loading_output,
            self.results_output,
            self.selector_container
        ])
     
    def browse_file(self, b):
        """Handle file browsing (placeholder)"""
        # In a real implementation, this would open a file dialog
        print("📂 File browser not implemented in Colab. Please enter path manually.")
     
    def extract_keyphrases_batch(self, sentences, top_k=5, threshold=0.1):
        """Extract keyphrases using BERT-KPE"""
        if not self.kpe_extractor:
            raise Exception("Keyphrase extractor not loaded")
         
        print(f"🔑 Extracting up to {top_k} keyphrases from {len(sentences)} sentences...")
         
        raw_kp_lists = self.kpe_extractor.batch_extract_keyphrases(
            texts=sentences,
            top_k=top_k,
            threshold=threshold
        )
         
        results = []
        for i, (sentence, kp_list) in enumerate(zip(sentences, raw_kp_lists)):
            # Update progress
            progress = int((i + 1) / len(sentences) * 50)  # First 50% for extraction
            self.progress_bar.value = progress
             
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
     
    def run_keyphrase_pipeline(self, pdf_path, max_sentences=50, top_k=5, threshold=0.1):
        """Main keyphrase pipeline"""
        from helper import (
            extract_text_from_pdf_robust,
            preprocess_pdf_text,
            extract_and_filter_sentences,
            classify_sentence_bert,
            classify_sentence_similarity,
            determine_dual_consensus,
            analyze_full_text_coreferences
        )
         
        print("🚀 Starting Keyphrase Analysis Pipeline")
        print("=" * 70)
        
        # Show progress bar
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 0
         
        # Extract and preprocess text
        self.progress_bar.description = "Extracting PDF..."
        raw = extract_text_from_pdf_robust(pdf_path)
        self.progress_bar.value = 10
         
        self.progress_bar.description = "Preprocessing..."
        clean = preprocess_pdf_text(raw)
        self.progress_bar.value = 20
         
        # Extract coreference chains
        self.progress_bar.description = "Analyzing coreferences..."
        print("\n🔍 Analyzing coreference chains...")
        chains_result = analyze_full_text_coreferences(clean)
        all_chains = chains_result["chains"]
        self.progress_bar.value = 30
         
        # Split sentences
        self.progress_bar.description = "Filtering sentences..."
        sents = extract_and_filter_sentences(clean)
        if len(sents) > max_sentences:
            print(f"⚠️  Limiting to first {max_sentences} sentences")
            sents = sents[:max_sentences]
        self.progress_bar.value = 40
         
        # Extract keyphrases
        self.progress_bar.description = "Extracting keyphrases..."
        print(f"\n🔑 Extracting keyphrases from {len(sents)} sentences...")
        kp_results = self.extract_keyphrases_batch(sents, top_k=top_k, threshold=threshold)
        self.progress_bar.value = 70
         
        # Process each sentence
        self.progress_bar.description = "Classifying sentences..."
        results = []
        
        for i, s in enumerate(sents):
            # Update progress
            progress = 70 + int((i + 1) / len(sents) * 25)  # Last 25% for classification
            self.progress_bar.value = progress
             
            # Classification
            b_lab, b_conf = classify_sentence_bert(s)
            si_lab, si_conf = classify_sentence_similarity(s)
            cons = determine_dual_consensus(b_lab, b_conf, si_lab, si_conf)
             
            # Find keyphrases for this sentence
            sentence_kps = []
            for kp_entry in kp_results:
                if kp_entry['sentence'] == s:
                    sentence_kps = [kp['phrase'] for kp in kp_entry['keyphrases']]
                    break
             
            # Placeholder for keyphrase-coreference analysis
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
         
        self.progress_bar.value = 100
        self.progress_bar.description = "Complete!"
         
        return {
            "results": results,
            "full_text": clean,
            "coref_chains": all_chains,
            "keyphrase_summary": {
                "total_sentences_with_keyphrases": len(kp_results),
                "total_keyphrases": sum(len(entry['keyphrases']) for entry in kp_results)
            }
        }
     
    def run_analysis(self, b):
        """Run the keyphrase analysis"""
        with self.results_output:
            clear_output(wait=True)
             
            if not self.file_input.value:
                print("❌ Please enter a PDF file path")
                return
             
            if not self.kpe_extractor:
                print("❌ Keyphrase extractor not loaded. Please wait for model loading to complete.")
                return
             
            try:
                # Run pipeline
                self.results = self.run_keyphrase_pipeline(
                    self.file_input.value,
                    max_sentences=self.max_sentences.value,
                    top_k=self.top_k_phrases.value,
                    threshold=self.kp_threshold.value
                )
                 
                # Show summary
                from helper import print_explanation_summary, create_selector
                print_explanation_summary(self.results)
                 
                # Create interactive selector
                selector_widget = create_selector(self.results)
                self.selector_container.children = [selector_widget[0], selector_widget[1]]
                 
                # Enable export button
                self.export_btn.disabled = False
                 
                # Hide progress bar
                self.progress_bar.layout.visibility = 'hidden'
                 
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                self.progress_bar.layout.visibility = 'hidden'
     
    def reset_interface(self, b):
        """Reset the interface"""
        self.file_input.value = ""
        self.results = None
        self.export_btn.disabled = True
        self.progress_bar.layout.visibility = 'hidden'
        self.progress_bar.value = 0
         
        with self.results_output:
            clear_output()
         
        self.selector_container.children = []
     
    def export_results(self, b):
        """Export results (placeholder)"""
        if self.results:
            print("💾 Export functionality would save results to file")
        else:
            print("❌ No results to export")
     
    def get_interface(self):
        """Return the main interface widget"""
        return self.interface
 
# =============================================================================
# Span Tab Interface
# =============================================================================
 
class SpanTabInterface:
    def __init__(self):
        self.results = None
        self.setup_interface()
        self.span_iface = SpanAnalysisInterface()
        self.analyze_spans_with_coreference = (
           self.span_iface.analyze_spans_with_coreference
           )
        self.compute_span_importances = (
            self.span_iface.compute_span_importances
           )
     
    def setup_interface(self):
        """Create the span analysis interface"""
         
        # Header
        header = widgets.HTML("""
        <div style='background:#e8f5e8; padding:15px; border-radius:8px; margin-bottom:15px;'>
            <h3 style='margin:0; color:#2ca02c;'>🎯 IG Span-based SDG Analysis</h3>
            <p style='margin:5px 0 0 0; color:#666;'>
                Use Integrated Gradients to identify important spans with coreference analysis
            </p>
        </div>
        """)
         
        # File input section
        file_section = widgets.VBox([
            widgets.HTML("<h4>📁 Input Configuration</h4>"),
            widgets.HBox([
                widgets.Text(
                    placeholder='Enter PDF file path...',
                    description='PDF Path:',
                    layout=widgets.Layout(width='400px'),
                    style={'description_width': '80px'}
                ),
                widgets.Button(
                    description='📂 Browse',
                    button_style='info',
                    layout=widgets.Layout(width='80px')
                )
            ])
        ])
         
        self.file_input = file_section.children[1].children[0]
        browse_btn = file_section.children[1].children[1]
        browse_btn.on_click(self.browse_file)
         
        # Parameters section
        params_section = widgets.VBox([
            widgets.HTML("<h4>⚙️ Analysis Parameters</h4>"),
            widgets.HBox([
                widgets.IntSlider(
                    value=30, min=10, max=100, step=5,
                    description='Max Sentences:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                ),
                widgets.IntSlider(
                    value=4, min=1, max=8, step=1,
                    description='Max Span Length:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                )
            ]),
            widgets.HBox([
                widgets.IntSlider(
                    value=5, min=1, max=10, step=1,
                    description='Top K Spans:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                ),
                widgets.FloatSlider(
                    value=0.5, min=0.1, max=0.9, step=0.1,
                    description='Min Confidence:',
                    style={'description_width': '120px'},
                    layout=widgets.Layout(width='300px')
                )
            ])
        ])
         
        self.max_sentences = params_section.children[1].children[0]
        self.max_span_len = params_section.children[1].children[1]
        self.top_k_spans = params_section.children[2].children[0]
        self.min_confidence = params_section.children[2].children[1]
         
        # Run section
        run_section = widgets.VBox([
            widgets.HTML("<h4>🚀 Execute Analysis</h4>"),
            widgets.HBox([
                widgets.Button(
                    description='🚀 Run Span Analysis',
                    button_style='success',
                    layout=widgets.Layout(width='250px', height='40px')
                ),
                widgets.Button(
                    description='🔄 Reset',
                    button_style='warning',
                    layout=widgets.Layout(width='100px', height='40px')
                ),
                widgets.Button(
                    description='💾 Export Results',
                    button_style='info',
                    layout=widgets.Layout(width='150px', height='40px'),
                    disabled=True
                )
            ])
        ])
         
        self.run_btn = run_section.children[1].children[0]
        self.reset_btn = run_section.children[1].children[1]
        self.export_btn = run_section.children[1].children[2]
         
        self.run_btn.on_click(self.run_analysis)
        self.reset_btn.on_click(self.reset_interface)
        self.export_btn.on_click(self.export_results)
         
        # Output areas
        self.results_output = widgets.Output()
        self.selector_container = widgets.VBox([])
        self.output = self.results_output 
        # Progress bar
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Progress:',
            bar_style='success',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
         
        # Main container
        self.interface = widgets.VBox([
            header,
            file_section,
            params_section,
            run_section,
            self.progress_bar,
            self.output,
            self.selector_container
        ])
     
    def browse_file(self, b):
        """Handle file browsing (placeholder)"""
        print("📂 File browser not implemented in Colab. Please enter path manually.")
     
    def run_span_pipeline(pdf_path,
                 agree_thresh=0.1, disagree_thresh=0.2, min_conf=0.5, max_sentences=50):
        """Main pipeline orchestrator with enhanced span-coreference logic"""
        print("🚀 Starting SDG Classification & Span Analysis Pipeline")
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
            cons = determine_dual_consensus(b_lab, b_conf, si_lab, si_conf,
                                        agree_thresh, disagree_thresh, min_conf)

            # Extract important spans using MASK
            span_scores, tokens, offsets = self.compute_span_importances(s, b_lab)
            top_spans = sorted(span_scores, key=lambda x: -x['importance'])[:top_k]

            #Extract span importances IG
            #span_scores, tokens, offsets = compute_span_importances_ig(s, b_lab)
            #top_spans = sorted(span_scores, key=lambda x: -x['importance'])[:5]

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

        print(f"\n✅ Pipeline completed successfully!")
        print(f"   📊 Processed {len(results)} sentences")
        print(f"   🔗 Found {len(all_chains)} coreference chains")
        print(f"   🎯 Analyzed spans with coreference mapping")

        return {
            "results": results,
            "full_text": clean,
            "coref_chains": all_chains,
            "span_summary": {
                "total_sentences_with_spans": len([r for r in results if r['span_analysis']]),
                "total_spans": sum(len(r['span_analysis']) for r in results)
                }
            }
         
        # Show progress bar as demo
        self.progress_bar.layout.visibility = 'visible'
        for i in range(101):
            self.progress_bar.value = i
        self.progress_bar.layout.visibility = 'hidden'
    
    def run_analysis(self, b):
        # delegate to the SpanAnalysisInterface
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 0

        # Run the pipeline and capture everything in results_output
        with self.results_output:
            clear_output(wait=True)
            if not self.file_input.value:
                print("❌ Please enter a PDF file path")
                self.progress_bar.layout.visibility = 'hidden'
                return

            try:
                # Kick off pipeline
                self.progress_bar.description = "Starting Span Analysis..."
                self.progress_bar.value       =  5

            # Run the IG-span pipeline
            
                self.results = self.span_iface.run_span_pipeline(
                    self.file_input.value,
                    max_sentences=self.max_sentences.value,
                    max_span_len=self.max_span_len.value
                )
                
                self.progress_bar.value = 75

                 # Plot the consensus + coref bar charts
                from helper import print_explanation_summary, create_selector
                clear_output(wait=True)
                print_explanation_summary(self.results)
                self.progress_bar.value = 90
 
                # Add the sentence selector dropdown
                dd, out = create_selector(self.results)
                self.selector_container.children = [dd, out]
                
                self.progress_bar.value = 100
 
            except Exception as e:
                print(f"❌ Error during span analysis: {e}")
            finally:
                # hide the bar when done (success or failure)
                self.progress_bar.layout.visibility = 'hidden'

    def reset_interface(self, b):
        """Reset the interface"""
        self.file_input.value = ""
        self.results = None
        self.export_btn.disabled = True
        self.progress_bar.layout.visibility = 'hidden'
        self.progress_bar.value = 0
         
        with self.results_output:
            clear_output()
         
        self.selector_container.children = []
     
    def export_results(self, b):
        """Export results (placeholder)"""
        if self.results:
            print("💾 Export functionality would save results to file")
        else:
            print("❌ No results to export")
     
    def get_interface(self):
        """Return the main interface widget"""
        return self.interface
 
# =============================================================================
# Comparison Tab Interface
# =============================================================================
 
class ComparisonTabInterface:
    def __init__(self):
        self.setup_interface()
     
    def setup_interface(self):
        """Create the comparison interface"""
         
        # Header
        header = widgets.HTML("""
        <div style='background:#fff2e6; padding:15px; border-radius:8px; margin-bottom:15px;'>
            <h3 style='margin:0; color:#ff7f0e;'>📊 Compare Analysis Results</h3>
            <p style='margin:5px 0 0 0; color:#666;'>
                Side-by-side comparison of Keyphrase vs IG Span analysis results
            </p>
        </div>
        """)
         
        # Comparison controls
        controls = widgets.VBox([
            widgets.HTML("<h4>🔄 Load Results for Comparison</h4>"),
            widgets.HBox([
                widgets.Button(
                    description='📁 Load Keyphrase Results',
                    button_style='info',
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Button(
                    description='📁 Load Span Results',
                    button_style='success',
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Button(
                    description='📊 Generate Comparison',
                    button_style='primary',
                    layout=widgets.Layout(width='200px'),
                    disabled=True
                )
            ])
        ])
         
        # Results display
        comparison_output = widgets.Output()
         
        # Status
        status = widgets.HTML("""
        <div style='padding:20px; text-align:center; color:#666;'>
            <p>📋 No results loaded yet</p>
            <p>Run analysis in the other tabs first, then return here for comparison</p>
        </div>
        """)
         
        self.interface = widgets.VBox([
            header,
            controls,
            status,
            comparison_output
        ])
     
    def get_interface(self):
        """Return the main interface widget"""
        return self.interface
 
# =============================================================================
# Main execution
# =============================================================================
 
def launch_sdg_interface():
    """Launch the main tabbed SDG interface"""
    return TabbedSDGInterface()
 
# Usage:
# interface = launch_sdg_interface()