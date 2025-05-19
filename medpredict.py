 # medpredict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# -------------------------------
# 1. Simulate Patient Dataset
# -------------------------------
data = pd.DataFrame({
    'age': np.random.randint(20, 90, 1000),
    'num_prev_admissions': np.random.randint(0, 5, 1000),
    'length_of_stay': np.random.randint(1, 15, 1000),
    'comorbidity_score': np.random.randint(0, 10, 1000),
    'has_diabetes': np.random.randint(0, 2, 1000),
    'has_hypertension': np.random.randint(0, 2, 1000),
    'readmitted': np.random.randint(0, 2, 1000)  # Target variable
})