# sdg_launcher.py

import importlib
import os

# — Colab widgets imports
import ipywidgets as widgets
from IPython.display import display, clear_output

# — Your main tabbed interface
from main_tabbed_interface import TabbedSDGInterface  # :contentReference[oaicite:4]{index=4}

def launch_colab_interface():
    """
    Colab‑style launcher that restores the full tabbed interface
    (Keyphrase, IG Span, Compare) from main_tabbed_interface.py.
    """
    # Reload the module so that any code changes are picked up
    importlib.reload(importlib.import_module("main_tabbed_interface"))

    # Instantiate the TabbedSDGInterface
    # Its __init__ calls setup_interface() which will display the widget
    TabbedSDGInterface()  # :contentReference[oaicite:5]{index=5}

if __name__ == "__main__":
    try:
        # If in a notebook/Colab, launch the tabs
        get_ipython()
        launch_colab_interface()
    except NameError:
        # Otherwise you could hook in a Streamlit entrypoint here
        pass
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:43:21 2025

@author: niran
"""

