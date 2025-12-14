#!/usr/bin/env bash
# Unix helper to create venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Note: install PyTorch from https://pytorch.org for the correct CUDA/cpu build." 
