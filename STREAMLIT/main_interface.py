# =============================================================================
# main_interface.py - Main interface with widget selection
# =============================================================================

import ipywidgets as widgets
from IPython.display import display, clear_output
import importlib

class SDGAnalysisInterface:
    def __init__(self):
        self.current_results = None
        self.setup_interface()
    
    def setup_interface(self):
        """Create the main interface with HBox selection"""
        
        # Title
        title = widgets.HTML("<h2>🎯 SDG Classification & Explainability System</h2>")
        
        # Method selection buttons
        keyphrase_btn = widgets.Button(
            description='🔑 Keyphrase Analysis',
            button_style='info',
            layout=widgets.Layout(width='200px', height='60px'),
            tooltip='Extract and analyze keyphrases with coreference resolution'
        )
        
        span_btn = widgets.Button(
            description='🎯 IG Span Analysis', 
            button_style='success',
            layout=widgets.Layout(width='200px', height='60px'),
            tooltip='Use Integrated Gradients for span importance analysis'
        )
        
        # Method selection HBox
        method_box = widgets.HBox([
            keyphrase_btn,
            widgets.HTML("<div style='width:50px'></div>"),  # spacer
            span_btn
        ], layout=widgets.Layout(justify_content='center', margin='20px'))
        
        # Output area
        self.output = widgets.Output()
        
        # Results area (initially hidden)
        self.results_area = widgets.VBox([])
        
        # Bind button events
        keyphrase_btn.on_click(lambda b: self.launch_keyphrase_analysis())
        span_btn.on_click(lambda b: self.launch_span_analysis())
        
        # Main container
        self.main_container = widgets.VBox([
            title,
            method_box,
            self.output,
            self.results_area
        ])
        
        display(self.main_container)
    
    def launch_keyphrase_analysis(self):
        """Launch the keyphrase analysis pipeline"""
        with self.output:
            clear_output()
            print("🔑 Loading Keyphrase Analysis Module...")
            
            try:
                # Import the keyphrase module
                import keyphrase_analysis
                importlib.reload(keyphrase_analysis)  # Ensure fresh import
                
                # Create the interface
                kp_interface = keyphrase_analysis.KeyphraseAnalysisInterface()
                self.results_area.children = [kp_interface.get_interface()]
                
            except Exception as e:
                print(f"❌ Error loading keyphrase analysis: {e}")
    
    def launch_span_analysis(self):
        """Launch the IG span analysis pipeline"""
        with self.output:
            clear_output()
            print("🎯 Loading IG Span Analysis Module...")
            
            try:
                # Import the span module
                import span_analysis
                importlib.reload(span_analysis)  # Ensure fresh import
                
                # Create the interface
                span_interface = span_analysis.SpanAnalysisInterface()
                self.results_area.children = [span_interface.get_interface()]
                
            except Exception as e:
                print(f"❌ Error loading span analysis: {e}")
