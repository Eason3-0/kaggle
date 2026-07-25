# -*- coding: utf-8 -*-
"""
====================================================================
Step 4: Generate Predictions & Submission File
Store Sales Prediction — Kaggle Competition
====================================================================

This script will:
  1. Load test feature matrix from data/features/
  2. Load trained model from models/
  3. Predict sales for the test period (2017-08-16 to 2017-08-31)
  4. Clip negative predictions to 0
  5. Export submission.csv to submissions/ (matching sample_submission format)

Usage:
    cd code/
    python step4_generate_submission.py
"""

from config import *
from utils import *

# TODO: Implement Step 4

if __name__ == "__main__":
    print("Step 4: Generate Submission — NOT YET IMPLEMENTED")
    print(f"Will read from:  {MODELS_DIR}")
    print(f"Will write to:   {SUBMISSIONS_DIR}")
