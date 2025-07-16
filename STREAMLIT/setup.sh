#!/usr/bin/env bash
set -e

# Clone the BERT-KPE repository (if not already present)
if [ ! -d "BERT-KPE" ]; then
  git clone https://github.com/thunlp/BERT-KPE.git
fi

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_sci_sm

# Install coreference resolution model for spaCy
python -m coreferee install en

# Install Python dependencies
pip install -r requirements.txt
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:52:58 2025

@author: niran
"""

