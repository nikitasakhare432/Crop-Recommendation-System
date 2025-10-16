import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_crop_model():
    """
    Train a RandomForestClassifier for crop recommendation
    Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    """

    csv_path = 'Crop_recommendation.csv'

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        return False

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nUnique crops: {df['label'].nunique()}")
    print(f"Crops: {sorted(df['label'].unique())}")

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    model_path = 'crop_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved successfully as '{model_path}'")
    print("You can now run the Streamlit app using: streamlit run app.py")

    return True

if __name__ == "__main__":
    success = train_crop_model()
    if not success:
        print("\nTraining failed. Please ensure the dataset is available.")
