import ipywidgets as widgets
from IPython.display import display, clear_output
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any

from span_analysis import SpanAnalysisInterface
from keyphrase_analysis import KeyphraseAnalysisInterface
from helper import print_explanation_summary, create_selector, render_cluster_graph


class TabbedSDGInterface:
    def __init__(self):
        self.setup_interface()

    def setup_interface(self):
        """Create the main tabbed interface"""
        tab = widgets.Tab()
        keyphrase_container = widgets.VBox()
        span_container      = widgets.VBox()
        comparison_container= widgets.VBox()

        tab.children = [
            keyphrase_container,
            span_container,
            comparison_container
        ]
        tab.set_title(0, '🔑 Keyphrase Analysis')
        tab.set_title(1, '🎯 IG Span Analysis')
        tab.set_title(2, '📊 Compare Results')

        # Initialize each tab
        self.initialize_keyphrase_tab(keyphrase_container)
        self.initialize_span_tab(span_container)
        self.initialize_comparison_tab(comparison_container)

        self.main_interface = widgets.VBox([
            widgets.HTML("<h2>🎯 SDG Classification & Explainability System</h2>"),
            widgets.HTML("<p>Select a tab to choose your analysis method:</p>"),
            tab
        ])
        display(self.main_interface)

    def initialize_keyphrase_tab(self, container):
        loading_output = widgets.Output()
        container.children = [loading_output]
        with loading_output:
            print("🔑 Initializing Keyphrase Analysis Module...")
        try:
            # create and store the Keyphrase tab
            self.keyphrase_interface = KeyphraseTabInterface()
            container.children = [self.keyphrase_interface.get_interface()]
        except Exception as e:
            with loading_output:
                clear_output()
                print(f"❌ Error loading Keyphrase Analysis: {e}")
                print("Make sure keyphrase_analysis.py and helper.py are available")

    def initialize_span_tab(self, container):
        loading_output = widgets.Output()
        container.children = [loading_output]
        with loading_output:
            print("🎯 Initializing IG Span Analysis Module...")
        try:
            # create and store the Span tab
            self.span_interface = SpanTabInterface()
            container.children = [self.span_interface.get_interface()]
        except Exception as e:
            with loading_output:
                clear_output()
                print(f"❌ Error loading Span Analysis: {e}")
                print("Make sure span_analysis.py and helper.py are available")

    def initialize_comparison_tab(self, container):
        # now pass the already‑built tab interfaces into ComparisonTabInterface
        try:
            cmp_iface = ComparisonTabInterface(
                self.keyphrase_interface,
                self.span_interface
            )
            container.children = [cmp_iface.get_interface()]
        except Exception as e:
            container.children = [widgets.HTML(
                f"❌ Could not initialize Comparison tab:<br>{e}"
            )]

class KeyphraseTabInterface:
    def __init__(self):
        self.kp_iface = KeyphraseAnalysisInterface()
        self.setup_interface()
        self.load_models()

    def setup_interface(self):
        # Inputs
        self.file_input = widgets.Text(
            placeholder='Path to PDF file', description='PDF file:'
        )
        self.max_sentences = widgets.IntSlider(
            value=5, min=1, max=50, description='Max Sentences'
        )
        self.top_k_phrases = widgets.IntSlider(
            value=5, min=1, max=20, description='Top-K Phrases'
        )
        self.kp_threshold = widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.01, description='Threshold'
        )
        self.min_confidence = widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.01, description='Min Conf.'
        )

        # Controls & outputs
        self.run_btn = widgets.Button(
            description='🚀 Run Keyphrase Analysis', button_style='primary'
        )
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=100, description='Progress:'
        )
        self.results_output = widgets.Output()
        self.selector_container = widgets.VBox()
        self.cluster_dropdown = widgets.Dropdown(options=[], description='Cluster:')
        self.graph_output = widgets.Output()

        # Wire callbacks
        self.run_btn.on_click(self.run_analysis)
        self.cluster_dropdown.observe(self.on_cluster_selected, names='value')

        # Layout
        self.interface = widgets.VBox([
            self.file_input,
            self.max_sentences,
            self.top_k_phrases,
            self.kp_threshold,
            self.min_confidence,
            self.run_btn,
            self.progress_bar,
            self.results_output,
            self.selector_container,
            self.cluster_dropdown,
            self.graph_output
        ])

    def load_models(self):
        # Delegate model loading to the analysis interface
        self.kp_iface.load_models()
        with self.results_output:
            print("✅ Keyphrase models loaded")

    def run_analysis(self, b):
        # Show progress
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 0
        with self.results_output:
            clear_output(wait=True)
            if not self.file_input.value:
                print("❌ Please enter a PDF file path")
                self.progress_bar.layout.visibility = 'hidden'
                return

            try:
                self.progress_bar.value = 10
                # Delegate to lower-level pipeline
                self.results = self.kp_iface.run_keyphrase_pipeline(
                    self.file_input.value,
                    max_sentences=self.max_sentences.value,
                    top_k=self.top_k_phrases.value,
                    threshold=self.kp_threshold.value,
                    min_conf=self.min_confidence.value
                    )
                self.progress_bar.value = 60

                # Summary bars
                clear_output(wait=True)
                print_explanation_summary(self.results)
                self.progress_bar.value = 80

                # Sentence selector - STORE THE DROPDOWN
                dd, out = create_selector(self.results)
                self.sentence_dropdown = dd  # ADD THIS LINE
                self.selector_container.children = [dd, out]

                # Populate cluster dropdown
                self.cluster_dropdown.options = [
                    (", ".join(c['mentions']), c['chain_id'])
                    for c in self.results.get('clusters', [])
                    ]

                # Clear previous graph
                with self.graph_output:
                    clear_output()
                    self.progress_bar.value = 100

            except Exception as e:
                print(f"❌ Error during keyphrase analysis: {e}")

            finally:
                self.progress_bar.layout.visibility = 'hidden'

    def on_cluster_selected(self, change):
        cid = change['new']
        if not hasattr(self, 'sentence_dropdown') or self.sentence_dropdown.value is None:
            return
    
        # Get selected sentence index
        sent_idx = self.sentence_dropdown.value
    
        # Find the cluster and compute focus_idx
        focus_idx = None
        for c in self.results.get('clusters', []):
            if c['chain_id'] == cid:
                # Get the sentence text for this index
                sent_text = self.results['results'][sent_idx]['sentence']
                # Find this sentence in the cluster's sentences
            try:
                focus_idx = c['sentences'].index(sent_text)
                
            except ValueError:
                focus_idx = None
            
            break

        with self.graph_output:
            clear_output()
            render_cluster_graph(
                cluster_id=cid,
                clusters=self.results.get('clusters', []),
                cluster_graphs=self.results.get('cluster_graphs', {}),
                focus_idx=focus_idx
                )

    def get_interface(self):
        return self.interface


class SpanTabInterface:
    def __init__(self):
        self.span_iface = SpanAnalysisInterface()
        self.setup_interface()
        self.load_models()

    def setup_interface(self):
        # Inputs
        self.file_input = widgets.Text(
            placeholder='Path to PDF file', description='PDF file:'
        )
        self.max_sentences = widgets.IntSlider(
            value=5, min=1, max=50, description='Max Sentences'
        )
        self.max_span_len = widgets.IntSlider(
            value=3, min=1, max=10, description='Max Span Len'
        )

        # Controls & outputs
        self.run_btn = widgets.Button(
            description='🚀 Run IG Span Analysis', button_style='primary'
        )
        self.progress_bar = widgets.IntProgress(
            value=0, min=0, max=100, description='Progress:'
        )
        self.results_output = widgets.Output()
        self.selector_container = widgets.VBox()
        self.cluster_dropdown = widgets.Dropdown(options=[], description='Cluster:')
        self.graph_output = widgets.Output()

        # Wire callbacks
        self.run_btn.on_click(self.run_analysis)
        self.cluster_dropdown.observe(self.on_cluster_selected, names='value')

        # Layout
        self.interface = widgets.VBox([
            self.file_input,
            self.max_sentences,
            self.max_span_len,
            self.run_btn,
            self.progress_bar,
            self.results_output,
            self.selector_container,
            self.cluster_dropdown,
            self.graph_output
        ])

    def load_models(self):
        # Delegate model loading
        self.span_iface.load_models()
        with self.results_output:
            print("✅ Span analysis models loaded")

    def run_analysis(self, b):
        # Show progress
        self.progress_bar.layout.visibility = 'visible'
        self.progress_bar.value = 0
        with self.results_output:
            clear_output(wait=True)
            if not self.file_input.value:
                print("❌ Please enter a PDF file path")
                self.progress_bar.layout.visibility = 'hidden'
                return

            try:
                self.progress_bar.value = 10
                # Delegate to lower-level pipeline
                self.results = self.span_iface.run_span_pipeline(
                    self.file_input.value,
                    max_sentences=self.max_sentences.value,
                    max_span_len=self.max_span_len.value
                )
                self.progress_bar.value = 60

                # Summary bars
                clear_output(wait=True)
                print_explanation_summary(self.results)
                self.progress_bar.value = 80

                # Sentence selector
                dd, out = create_selector(self.results)
                self.sentence_dropdown = dd
                self.selector_container.children = [dd, out]

                # Populate cluster dropdown
                self.cluster_dropdown.options = [
                    (", ".join(c['mentions']), c['chain_id'])
                    for c in self.results.get('clusters', [])
                ]

                # Clear previous graph
                with self.graph_output:
                    clear_output()
                self.progress_bar.value = 100

            except Exception as e:
                print(f"❌ Error during span analysis: {e}")

            finally:
                self.progress_bar.layout.visibility = 'hidden'

    def on_cluster_selected(self, change):
        cid = change['new']
        # get the currently selected sentence index
        sent_idx = self.sentence_dropdown.value

        # find the cluster object and compute focus_idx
        focus_idx = None
        for c in self.results.get('clusters', []):
            if c['chain_id'] == cid:
                try:
                    sent_text = self.results['results'][sent_idx]['sentence']
                    focus_idx = c['sentences'].index(sent_text)
                except ValueError:
                    focus_idx = None
                break

        with self.graph_output:
            clear_output()
            render_cluster_graph(
                cluster_id=cid,
                clusters=self.results.get('clusters', []),
                cluster_graphs=self.results.get('cluster_graphs', {}),
                focus_idx=focus_idx
            )

    def get_interface(self):
        return self.interface


class ComparisonTabInterface:
    def __init__(self, kp_tab: KeyphraseTabInterface, span_tab: SpanTabInterface):
        # Future: implement side-by-side compare
        self.kp_tab = kp_tab
        self.span_tab = span_tab
        self.interface = widgets.Label("Comparison tab coming soon…")

    def get_interface(self):
        return self.interface



def launch_tabbed_ui():
    kp_tab = KeyphraseTabInterface()
    span_tab = SpanTabInterface()
    cmp_tab = ComparisonTabInterface(kp_tab, span_tab)

    tabs = widgets.Tab(children=[
        kp_tab.get_interface(),
        span_tab.get_interface(),
        cmp_tab.get_interface()
    ])
    tabs.set_title(0, "🔑 Keyphrase Analysis")
    tabs.set_title(1, "📊 IG Span Analysis")
    tabs.set_title(2, "🔍 Compare Results")

    display(tabs)

def launch_sdg_interface():
    return TabbedSDGInterface()
