import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import gc  # Garbage collector for memory management
import time

# 1. Load dataset
print("Loading dataset...")
symptoms_disease_df = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')

# 2. Data exploration
print("\nSymptoms-Disease Dataset Information:")
print(f"Shape: {symptoms_disease_df.shape}")
print("\nFirst 5 rows:")
print(symptoms_disease_df.head())

# Identify disease column (likely the first column)
disease_column = symptoms_disease_df.columns[0]
print(f"\nDisease column name: {disease_column}")

# Count occurrences of each disease
disease_counts = symptoms_disease_df[disease_column].value_counts()
print(f"\nTotal unique diseases: {len(disease_counts)}")
print("\nTop 10 diseases by frequency:")
print(disease_counts.head(10))

# Analyze symptom distribution
symptom_columns = symptoms_disease_df.drop(disease_column, axis=1).columns
symptom_counts = symptoms_disease_df[symptom_columns].sum().sort_values(ascending=False)
print(f"\nTotal symptoms features: {len(symptom_columns)}")
print("\nTop 10 most common symptoms:")
print(symptom_counts.head(10))

# 3. Prepare data for modeling
X = symptoms_disease_df.iloc[:, 1:]  # Features (symptoms)
y = symptoms_disease_df.iloc[:, 0]   # Target (disease)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData prepared for modeling:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Free up memory
del symptoms_disease_df
gc.collect()

# 4. Train improved RandomForest model
print("\n" + "="*80)
print("TRAINING RANDOM FOREST MODEL WITH IMPROVED PARAMETERS".center(80))
print("="*80)

# Use more trees and better parameters to improve performance
start_time = time.time()
model = RandomForestClassifier(
    n_estimators=100,       # Increased from 50 to 100
    max_depth=None,         # Allow full depth for better performance
    min_samples_split=5,    # Reduced from 10 to 5
    min_samples_leaf=1,     # Default value
    max_features='sqrt',    # Typical good value for classification
    bootstrap=True,         # Use bootstrap samples
    class_weight='balanced', # Account for class imbalance
    random_state=42,
    n_jobs=-1,              # Use all available cores
    verbose=1               # Show progress
)

# Train the model
print(f"\nTraining started at: {time.strftime('%H:%M:%S')}")
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed at: {time.strftime('%H:%M:%S')}")
print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# 5. Evaluate model with clear formatting
print("\n" + "="*80)
print("MODEL EVALUATION RESULTS".center(80))
print("="*80)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "*"*50)
print(f"MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)".center(50))
print("*"*50 + "\n")

# Get summary metrics
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print("Classification Report Summary:")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Weighted average precision: {report_dict['weighted avg']['precision']:.4f} ({report_dict['weighted avg']['precision']*100:.2f}%)")
print(f"Weighted average recall:    {report_dict['weighted avg']['recall']:.4f} ({report_dict['weighted avg']['recall']*100:.2f}%)")
print(f"Weighted average F1-score:  {report_dict['weighted avg']['f1-score']:.4f} ({report_dict['weighted avg']['f1-score']*100:.2f}%)")

# 6. Feature importance with better visualization
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS".center(80))
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': symptom_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 most important symptoms for prediction:")
top_features = feature_importance.head(20)
for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
    print(f"{i+1:2d}. {feature:<30} {importance:.6f} ({importance*100:.2f}%)")

# Generate visualizations
plt.figure(figsize=(12, 10))
plt.title('Top 20 Most Important Symptoms for Disease Prediction', fontsize=16)
plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("\nFeature importance visualization saved as 'feature_importance.png'")

# 7. Test the model with some example predictions
print("\n" + "="*80)
print("MODEL PREDICTION EXAMPLES".center(80))
print("="*80)

# Create prediction function
def predict_disease(symptoms_input, model, symptom_columns):
    """
    Predict disease based on symptoms input
    symptoms_input: list of symptom names present
    """
    # Create input dataframe
    input_data = pd.DataFrame(0, index=[0], columns=symptom_columns)
    
    # Set existing symptoms to 1
    for symptom in symptoms_input:
        if symptom in input_data.columns:
            input_data[symptom] = 1
    
    # Predict
    predicted_disease = model.predict(input_data)[0]
    
    # Get top 3 probabilities
    probabilities = model.predict_proba(input_data)[0]
    top_indices = probabilities.argsort()[-3:][::-1]
    top_diseases = [model.classes_[i] for i in top_indices]
    top_probs = [probabilities[i] for i in top_indices]
    
    return predicted_disease, top_diseases, top_probs

# Choose 3 example test cases with common symptoms
test_cases = [
    ['headache', 'fever', 'cough'],
    ['sharp abdominal pain', 'nausea', 'vomiting'],
    ['back pain', 'shortness of breath', 'dizziness']
]

print("\nExample Predictions:")
print("-" * 80)
for i, symptoms in enumerate(test_cases):
    print(f"Case {i+1}: Patient with symptoms: {', '.join(symptoms)}")
    try:
        predicted_disease, top_diseases, top_probs = predict_disease(symptoms, model, symptom_columns)
        print(f"  Predicted disease: {predicted_disease}")
        print("  Top 3 most likely diseases:")
        for j, (disease, prob) in enumerate(zip(top_diseases, top_probs)):
            print(f"    {j+1}. {disease:<30} Confidence: {prob:.4f} ({prob*100:.2f}%)")
    except Exception as e:
        print(f"  Error in prediction: {e}")
    print("-" * 80)

# 8. Save model and important data
print("\n" + "="*80)
print("SAVING MODEL AND DATA".center(80))
print("="*80)

with open('disease_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as 'disease_prediction_model.pkl'")

# Save symptoms list and top diseases
with open('symptoms_list.pkl', 'wb') as f:
    pickle.dump(list(symptom_columns), f)
print("Symptoms list saved as 'symptoms_list.pkl'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved as 'feature_importance.csv'")

print("\n" + "="*80)
print("ALL PROCESSING COMPLETED SUCCESSFULLY!".center(80))
print("="*80)

# Optional: Create a confusion matrix for top 10 diseases
# Note: This can be very large with 773 classes, so we limit to top diseases
try:
    top_10_diseases = disease_counts.head(10).index
    mask = np.isin(y_test, top_10_diseases) & np.isin(y_pred, top_10_diseases)
    y_test_top10 = y_test[mask]
    y_pred_top10 = y_pred[mask]
    
    if len(y_test_top10) > 0:
        cm = confusion_matrix(y_test_top10, y_pred_top10, labels=top_10_diseases)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_10_diseases, yticklabels=top_10_diseases)
        plt.title('Confusion Matrix for Top 10 Diseases')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_top10.png', dpi=300)
        print("Confusion matrix for top 10 diseases saved as 'confusion_matrix_top10.png'")
except Exception as e:
    print(f"Could not create confusion matrix: {e}")