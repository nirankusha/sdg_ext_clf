# =============================================================================
# launcher.py - Simple launcher for the SDG Analysis System
# =============================================================================

# Run this in your Jupyter notebook to launch the interface

"""
🎯 SDG Classification & Explainability System Launcher

This script provides a tabbed interface for running different SDG analysis methods:
- 🔑 Keyphrase Analysis: BERT-KPE based approach
- 🎯 IG Span Analysis: Integrated Gradients based approach  
- 📊 Compare Results: Side-by-side comparison

Prerequisites:
1. Make sure all required files are in place:
   - helper.py (shared functions)
   - main_tabbed_interface.py (this interface)
   
2. Install required packages:
   pip install transformers sentence-transformers captum PyPDF2 spacy coreferee nltk matplotlib ipywidgets

3. Download spaCy model:
   python -m spacy download en_core_web_sm
   python -m coreferee install en
"""

# Import required libraries
import sys
import importlib
from pathlib import Path

def check_requirements():
    """Check if all requirements are available"""
    
    print("🔍 Checking requirements...")
    
    # Check required packages
    required_packages = [
        'torch', 'transformers', 'sentence_transformers', 
        'captum', 'PyPDF2', 'spacy', 'coreferee', 
        'nltk', 'matplotlib', 'ipywidgets'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check required files
    required_files = ['helper.py']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            print(f"  ❌ {file}")
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Please make sure all required files are in the current directory.")
        return False
    
    print("\n✅ All requirements satisfied!")
    return True

def launch_interface():
    """Launch the main SDG analysis interface"""
    
    # Check requirements first
    if not check_requirements():
        return None
    
    try:
        # Import the interface classes (assuming they're in the same file or notebook)
        from main_tabbed_interface import TabbedSDGInterface

        
        print("🚀 Launching SDG Analysis Interface...")
        interface = TabbedSDGInterface()
        print("✅ Interface launched successfully!")
        
        return interface
        
    except ImportError as e:
        print(f"❌ Could not import interface classes: {e}")
        print("Make sure the TabbedSDGInterface class is defined in this notebook.")
        return None
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        return None

# Quick launch function for notebook use
def quick_launch():
    
    """Quick launch with minimal setup"""
    print("🎯 SDG Analysis System - Quick Launch")
    print("=" * 50)
    from main_tabbed_interface import TabbedSDGInterface

    try:
        # Try to import and launch directly
        interface = TabbedSDGInterface()
        return interface
    except Exception as e:
        print(f"❌ Quick launch failed: {e}")
        print("Falling back to full requirement check...")
        return launch_interface()

# =============================================================================
# Usage Examples
# =============================================================================

def demo_usage():
    """Show usage examples"""
    
    print("""
    📚 USAGE EXAMPLES:
    
    # In your Jupyter notebook cell:
    
    # Option 1: Quick launch (recommended)
    interface = quick_launch()
    
    # Option 2: Full launch with requirement checking
    interface = launch_interface()
    
    # Option 3: Manual instantiation
    interface = TabbedSDGInterface()
    
    ==========================================
    
    🎯 TAB DESCRIPTIONS:
    
    📑 Tab 1 - Keyphrase Analysis:
    - Upload PDF file
    - Set parameters (max sentences, top K phrases, etc.)
    - Run BERT-KPE based analysis
    - Interactive sentence exploration
    
    📑 Tab 2 - IG Span Analysis:  
    - Upload PDF file
    - Set parameters (max sentences, span length, etc.)
    - Run Integrated Gradients analysis
    - Interactive span exploration
    
    📑 Tab 3 - Compare Results:
    - Load results from both methods
    - Side-by-side comparison
    - Comparative visualizations
    
    ==========================================
    
    💡 TIPS:
    - Start with Tab 1 or 2 to run analysis
    - Use smaller sentence limits for faster processing
    - Check the progress bars during analysis
    - Results are interactive - click dropdown to explore
    - Export functionality saves results for later use
    """)

if __name__ == "__main__":
    # Show usage information
    demo_usage()
    
    # Launch interface
    interface = quick_launch()

# =============================================================================
# Notebook Cell Examples
# =============================================================================

"""
# COPY THESE TO JUPYTER NOTEBOOK CELLS:

# Cell 1: Install packages (run once)
!pip install transformers sentence-transformers captum PyPDF2 spacy coreferee nltk matplotlib ipywidgets
!python -m spacy download en_core_web_sm
!python -m coreferee install en

# Cell 2: Import and setup (run once per session)
# Paste the TabbedSDGInterface code here or import from file

# Cell 3: Launch interface (run to start)
interface = TabbedSDGInterface()

# Cell 4: Demo usage (optional)
demo_usage()
"""