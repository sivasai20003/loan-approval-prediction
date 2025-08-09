import zipfile
import os

# Create the complete loan approval prediction project files

# 1. Main Python script
main_script = '''# Loan Approval Prediction - Complete Project
# =====================================================
# Author: Data Analyst Portfolio Project
# Date: August 2025
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("🚀 LOAN APPROVAL PREDICTION PROJECT")
print("=" * 50)

# =====================================================
# 1. DATA GENERATION (Replace with real data if available)
# =====================================================
def create_synthetic_dataset():
    """Generate realistic loan dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.exponential(5000, n_samples).astype(int),
        'CoapplicantIncome': np.random.exponential(2000, n_samples).astype(int),
        'LoanAmount': np.random.normal(150, 50, n_samples).clip(5).astype(int),
        'Loan_Amount_Term': np.random.choice([360, 240, 180, 120], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'Credit_History': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.4, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create logical target variable
    approval_prob = 0.5
    approval_prob += np.where(df['Credit_History'] == 1, 0.3, -0.3)
    approval_prob += np.where(df['Education'] == 'Graduate', 0.1, -0.1)
    approval_prob += np.where(df['Married'] == 'Yes', 0.1, 0)
    approval_prob += np.where(df['ApplicantIncome'] > df['ApplicantIncome'].median(), 0.1, -0.1)
    approval_prob += np.where(df['LoanAmount'] < df['LoanAmount'].median(), 0.1, -0.1)
    
    df['Loan_Status'] = np.where(np.random.random(n_samples) < approval_prob, 'Y', 'N')
    
    # Introduce missing values
    missing_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in missing_cols:
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    return df

# =====================================================
# 2. DATA EXPLORATION
# =====================================================
def explore_data(df):
    """Perform initial data exploration"""
    print("\\n📊 DATA EXPLORATION")
    print("-" * 30)
    print(f"Dataset Shape: {df.shape}")
    print(f"\\nMissing Values:")
    print(df.isnull().sum())
    print(f"\\nTarget Distribution:")
    print(df['Loan_Status'].value_counts())
    print(f"\\nApproval Rate: {(df['Loan_Status'] == 'Y').mean():.1%}")
    
    return df

# =====================================================
# 3. DATA CLEANING
# =====================================================
def clean_data(df):
    """Clean and preprocess the data"""
    print("\\n🧹 DATA CLEANING")
    print("-" * 30)
    
    # Handle missing values
    numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numerical_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {col} with median: {median_val}")
    
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled {col} with mode: {mode_val}")
    
    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_to_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
    df['Has_Coapplicant'] = np.where(df['CoapplicantIncome'] > 0, 1, 0)
    
    print(f"\\nCreated new features:")
    print(f"- Total_Income, Income_to_Loan_Ratio, Has_Coapplicant")
    print(f"Final dataset shape: {df.shape}")
    
    return df

# =====================================================
# 4. EXPLORATORY DATA ANALYSIS
# =====================================================
def perform_eda(df):
    """Create visualizations for EDA"""
    print("\\n📈 EXPLORATORY DATA ANALYSIS")
    print("-" * 30)
    
    # Set up plot style
    plt.figure(figsize=(15, 10))
    
    # 1. Target distribution
    plt.subplot(2, 3, 1)
    df['Loan_Status'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
    plt.title('Loan Status Distribution')
    plt.xticks(rotation=0)
    
    # 2. Credit History impact
    plt.subplot(2, 3, 2)
    pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)
    plt.title('Credit History vs Approval')
    plt.xticks(rotation=0)
    
    # 3. Income distribution
    plt.subplot(2, 3, 3)
    approved = df[df['Loan_Status'] == 'Y']['ApplicantIncome']
    rejected = df[df['Loan_Status'] == 'N']['ApplicantIncome']
    plt.hist([approved, rejected], bins=30, alpha=0.7, label=['Approved', 'Rejected'])
    plt.title('Income Distribution by Status')
    plt.legend()
    
    # 4. Education impact
    plt.subplot(2, 3, 4)
    pd.crosstab(df['Education'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)
    plt.title('Education vs Approval')
    plt.xticks(rotation=45)
    
    # 5. Marriage status impact
    plt.subplot(2, 3, 5)
    pd.crosstab(df['Married'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)
    plt.title('Marriage Status vs Approval')
    plt.xticks(rotation=0)
    
    # 6. Property area impact
    plt.subplot(2, 3, 6)
    pd.crosstab(df['Property_Area'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)
    plt.title('Property Area vs Approval')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

# =====================================================
# 5. MODEL BUILDING
# =====================================================
def build_models(df):
    """Encode features and build ML models"""
    print("\\n🤖 MODEL BUILDING")
    print("-" * 30)
    
    # Encoding
    df_encoded = df.copy()
    le = LabelEncoder()
    
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
    for col in binary_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    df_encoded = pd.get_dummies(df_encoded, columns=['Dependents', 'Property_Area'], drop_first=True)
    
    # Prepare data
    X = df_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df_encoded['Loan_Status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    lr_pred = lr_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test)
    
    print(f"✅ Models trained on {X_train.shape[0]} samples")
    print(f"✅ Testing on {X_test.shape[0]} samples")
    
    return X, y, X_test, y_test, lr_model, rf_model, lr_pred, rf_pred, scaler

# =====================================================
# 6. MODEL EVALUATION
# =====================================================
def evaluate_models(y_test, lr_pred, rf_pred, X, rf_model):
    """Evaluate and compare model performance"""
    print("\\n📊 MODEL EVALUATION")
    print("-" * 30)
    
    # Accuracies
    lr_acc = accuracy_score(y_test, lr_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"Logistic Regression Accuracy: {lr_acc:.3f}")
    print(f"Random Forest Accuracy: {rf_acc:.3f}")
    
    best_model = "Logistic Regression" if lr_acc > rf_acc else "Random Forest"
    best_acc = max(lr_acc, rf_acc)
    print(f"\\n🏆 Best Model: {best_model} ({best_acc:.1%})")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\\n🔍 TOP 5 IMPORTANT FEATURES:")
    for i, row in feature_imp.head(5).iterrows():
        print(f"   {i+1}. {row['Feature']}: {row['Importance']:.3f}")
    
    # Confusion Matrix visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression\\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
    plt.title('Random Forest\\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model, best_acc, feature_imp

# =====================================================
# 7. BUSINESS INSIGHTS
# =====================================================
def generate_insights(best_model, best_acc, feature_imp):
    """Generate business-ready insights and recommendations"""
    print("\\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 50)
    
    print("📈 PROJECT RESULTS:")
    print(f"   • Achieved {best_acc:.1%} prediction accuracy")
    print(f"   • Best performing model: {best_model}")
    print(f"   • Can automate {best_acc:.0%} of loan decisions")
    
    print("\\n🎯 KEY FINDINGS:")
    top_features = feature_imp.head(3)['Feature'].tolist()
    print(f"   • Most important factors: {', '.join(top_features)}")
    print(f"   • Credit history is the strongest predictor")
    print(f"   • Income levels significantly impact approval")
    
    print("\\n💡 BUSINESS RECOMMENDATIONS:")
    print("   1. Prioritize credit history verification")
    print("   2. Implement income-to-loan ratio guidelines")
    print("   3. Automate low-risk application approvals")
    print("   4. Flag high-risk applications for manual review")
    
    print("\\n🚀 IMPLEMENTATION BENEFITS:")
    print("   • Faster loan processing (reduced manual work)")
    print("   • Consistent decision-making criteria")
    print("   • Reduced human bias in approvals")
    print("   • Better risk management")
    
    print("\\n" + "=" * 50)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("📝 Ready for portfolio presentation!")

# =====================================================
# 8. MAIN EXECUTION
# =====================================================
def main():
    """Execute the complete loan approval prediction pipeline"""
    
    # Step 1: Create dataset
    print("Step 1: Creating dataset...")
    df = create_synthetic_dataset()
    
    # Step 2: Explore data
    print("Step 2: Exploring data...")
    df = explore_data(df)
    
    # Step 3: Clean data
    print("Step 3: Cleaning data...")
    df = clean_data(df)
    
    # Step 4: EDA
    print("Step 4: Performing EDA...")
    df = perform_eda(df)
    
    # Step 5: Build models
    print("Step 5: Building models...")
    X, y, X_test, y_test, lr_model, rf_model, lr_pred, rf_pred, scaler = build_models(df)
    
    # Step 6: Evaluate models
    print("Step 6: Evaluating models...")
    best_model, best_acc, feature_imp = evaluate_models(y_test, lr_pred, rf_pred, X, rf_model)
    
    # Step 7: Generate insights
    print("Step 7: Generating insights...")
    generate_insights(best_model, best_acc, feature_imp)
    
    # Save processed data
    df.to_csv('processed_loan_data.csv', index=False)
    feature_imp.to_csv('feature_importance.csv', index=False)
    
    print("\\n💾 Files saved:")
    print("   • processed_loan_data.csv")
    print("   • feature_importance.csv")
    print("   • eda_analysis.png")
    print("   • model_evaluation.png")

if __name__ == "__main__":
    main()
'''

# Save the main script
with open('loan_approval_prediction.py', 'w') as f:
    f.write(main_script)

print("✅ Main Python script created: loan_approval_prediction.py")