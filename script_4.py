# Create simplified Python script for quick execution
simple_script = '''#!/usr/bin/env python3
# Quick Loan Approval Prediction Demo
# Run this for immediate results

import pandas as pd, numpy as np, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings("ignore")

print("🚀 LOAN APPROVAL PREDICTION - QUICK DEMO")
print("=" * 50)

# Generate data
np.random.seed(42); N = 1000
df = pd.DataFrame({
    "Gender": np.random.choice(["Male","Female"], N, p=[.6,.4]),
    "Married": np.random.choice(["Yes","No"], N, p=[.7,.3]),
    "Education": np.random.choice(["Graduate","Not Graduate"], N, p=[.7,.3]),
    "ApplicantIncome": np.random.exponential(5000,N).astype(int),
    "CoapplicantIncome": np.random.exponential(2000,N).astype(int),
    "LoanAmount": np.random.normal(150,50,N).clip(5).astype(int),
    "Credit_History": np.random.choice([1,0], N, p=[.8,.2]),
})

# Create target
prob = 0.5 + np.where(df.Credit_History==1, 0.3,-0.3) + np.where(df.Education=="Graduate",0.1,-0.1)
df["Loan_Status"] = np.where(np.random.rand(N) < prob, "Y", "N")

# Feature engineering
df["Total_Income"] = df.ApplicantIncome + df.CoapplicantIncome
df["Income_Loan_Ratio"] = df.Total_Income / df.LoanAmount

print(f"📊 Dataset: {df.shape[0]} loans, {df.shape[1]} features")
print(f"📈 Approval Rate: {(df.Loan_Status=='Y').mean():.1%}")

# Encode and split
le = LabelEncoder()
for col in ["Gender","Married","Education","Loan_Status"]:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status",axis=1)
y = df["Loan_Status"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

# Train models
scaler = StandardScaler().fit(X_train)
lr = LogisticRegression().fit(scaler.transform(X_train), y_train)
rf = RandomForestClassifier(n_estimators=50,random_state=42).fit(X_train,y_train)

# Evaluate
lr_acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"\\n🎯 RESULTS:")
print(f"Logistic Regression: {lr_acc:.1%}")
print(f"Random Forest: {rf_acc:.1%}")
print(f"Best Model: {'LR' if lr_acc>rf_acc else 'RF'}")

# Feature importance
imp = pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False)
print(f"\\n🔍 Top 3 Features:")
for i, (feat, val) in enumerate(imp.head(3).items(), 1):
    print(f"{i}. {feat}: {val:.1%}")

print(f"\\n✅ Demo completed! Full project available in other files.")
'''

with open('quick_demo.py', 'w') as f:
    f.write(simple_script)

print("✅ quick_demo.py created")