import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap

# 1. Create a folder named 'assets' to store the pictures
os.makedirs('assets', exist_ok=True)
print("Generating charts and saving to 'assets' folder...")

# ==========================================
# CHART 1: Model Comparison Bar Chart
# ==========================================
print("1. Generating Model Comparison Chart...")
models = ['Decision Tree', 'Random Forest', 'XGBoost']
accuracies = [0.89, 0.95, 0.97]  # You can update these numbers based on your actual results

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette='Blues_d')
plt.title('Model Performance Comparison (Accuracy)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=12)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('assets/model_comparison.png', dpi=300)
plt.close() # Closes the plot so the next one can start cleanly

# ==========================================
# CHART 2: Feature Importance Plot
# ==========================================
print("2. Generating Feature Importance Plot...")
# Load your trained Random Forest model artifact
with open('src/models/artifacts/randomforest.pkl', 'rb') as f:
    rf_artifact = pickle.load(f)

rf_model = rf_artifact['model']
feature_names = rf_artifact['feature_names']

# Extract and sort features
importances = rf_model.feature_importances_
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top_features = feature_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 10 Most Important Lexical Features', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.savefig('assets/feature_importance.png', dpi=300)
plt.close()

# ==========================================
# LOAD TEST DATA FOR CHARTS 3 & 4
# ==========================================
print("Loading test data from data/test.csv...")
# Load the test split created by your ingest script
test_df = pd.read_csv('data/test.csv')
X_test = test_df[feature_names].fillna(0)
y_test = test_df['label']

# ==========================================
# CHART 3: Confusion Matrix
# ==========================================
print("3. Generating Confusion Matrix...")
# Predict on the test data to see what the model got right vs wrong
y_pred_idx = rf_model.predict(X_test)
y_pred = rf_artifact['label_encoder'].inverse_transform(y_pred_idx)

# Create the matrix
classes = sorted(list(set(y_test) | set(y_pred)))
cm = confusion_matrix(y_test, y_pred, labels=classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=300)
plt.close()
# ==========================================
# CHART 4: SHAP Summary Plot
# ==========================================
print("4. Generating SHAP Plot (This might take a minute)...")
try:
    # Use the Random Forest model already loaded above
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    # For multi-class models, shap_values is a list of arrays (one for each class)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        
    plt.title('SHAP Impact on Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP plot generated successfully!")
except Exception as e:
    print(f"Skipping SHAP due to error: {e}")
# ==========================================