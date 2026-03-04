import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── STEP 1: Clean the data ──────────────────────────────────
# Fix TotalCharges — it has some empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID — not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# ── STEP 2: Encode categorical columns ─────────────────────
# Machine learning models need numbers, not text
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ── STEP 3: Split into features and target ──────────────────
X = df.drop('Churn', axis=1)   # Everything except Churn
y = df['Churn']                 # Just Churn (what we predict)

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# ── STEP 4: Train the model ─────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── STEP 5: Evaluate the model ──────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Not Churned', 'Churned']))

# ── STEP 6: Feature Importance ──────────────────────────────
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print("\nTop 10 factors that predict churn:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'][::-1], 
         importance['Importance'][::-1], 
         color='#2e6da4')
plt.title("Top 10 Factors That Predict Customer Churn", 
          fontsize=14, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Feature importance chart saved!")