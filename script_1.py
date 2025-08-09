# Create Jupyter Notebook version
notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Loan Approval Prediction Project\\n",
    "**Author:** Your Name  \\n",
    "**Date:** August 2025  \\n",
    "**Objective:** Build a machine learning model to predict loan approval decisions\\n",
    "\\n",
    "## Project Overview\\n",
    "This project demonstrates end-to-end data analysis skills including data cleaning, exploratory analysis, feature engineering, machine learning modeling, and business insights generation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from sklearn.model_selection import train_test_split\\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\\n",
    "from sklearn.ensemble import RandomForestClassifier\\n",
    "from sklearn.linear_model import LogisticRegression\\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "print('📚 Libraries imported successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Generation (Replace with Real Dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create synthetic dataset (replace with real Kaggle data)\\n",
    "np.random.seed(42)\\n",
    "n_samples = 1000\\n",
    "\\n",
    "data = {\\n",
    "    'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],\\n",
    "    'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),\\n",
    "    'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),\\n",
    "    'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),\\n",
    "    'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),\\n",
    "    'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),\\n",
    "    'ApplicantIncome': np.random.exponential(5000, n_samples).astype(int),\\n",
    "    'CoapplicantIncome': np.random.exponential(2000, n_samples).astype(int),\\n",
    "    'LoanAmount': np.random.normal(150, 50, n_samples).clip(5).astype(int),\\n",
    "    'Loan_Amount_Term': np.random.choice([360, 240, 180, 120], n_samples, p=[0.7, 0.15, 0.1, 0.05]),\\n",
    "    'Credit_History': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]),\\n",
    "    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.4, 0.2])\\n",
    "}\\n",
    "\\n",
    "df = pd.DataFrame(data)\\n",
    "\\n",
    "# Create realistic target variable\\n",
    "approval_prob = 0.5\\n",
    "approval_prob += np.where(df['Credit_History'] == 1, 0.3, -0.3)\\n",
    "approval_prob += np.where(df['Education'] == 'Graduate', 0.1, -0.1)\\n",
    "approval_prob += np.where(df['Married'] == 'Yes', 0.1, 0)\\n",
    "approval_prob += np.where(df['ApplicantIncome'] > df['ApplicantIncome'].median(), 0.1, -0.1)\\n",
    "\\n",
    "df['Loan_Status'] = np.where(np.random.random(n_samples) < approval_prob, 'Y', 'N')\\n",
    "\\n",
    "print(f'Dataset created: {df.shape}')\\n",
    "df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''

# Save the notebook
with open('Loan_Approval_Prediction.ipynb', 'w') as f:
    f.write(notebook_content)

print("✅ Jupyter Notebook created: Loan_Approval_Prediction.ipynb")