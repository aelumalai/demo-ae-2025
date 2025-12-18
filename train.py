"""
Training script for Module 5 Assignment.

This script demonstrates how to train a machine learning model
using the utility functions provided in the src/ directory.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import utility functions
from src.data_preprocessing import scale_features, handle_missing_values
from src.model_utils import evaluate_model, plot_confusion_matrix, print_classification_report


def main():
    """Main training function."""
    print("=" * 60)
    print("Module 5 Assignment - Model Training")
    print("=" * 60)
    
    # Generate sample data for demonstration
    # Replace this with your actual data loading code
    print("\n1. Loading data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    print("\n3. Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("Features scaled successfully")
    
    # Train model
    print("\n4. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("Model training completed")
    
    # Make predictions
    print("\n5. Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    print("\n6. Evaluating model performance...")
    print("\nTraining Set Performance:")
    train_metrics = evaluate_model(y_train, y_train_pred, model_name='Random Forest (Train)')
    
    print("\nTest Set Performance:")
    test_metrics = evaluate_model(y_test, y_test_pred, model_name='Random Forest (Test)')
    
    # Print classification report
    print_classification_report(y_test, y_test_pred, target_names=['Class 0', 'Class 1'])
    
    # Plot confusion matrix
    print("\n7. Generating visualizations...")
    plot_confusion_matrix(
        y_test, 
        y_test_pred, 
        labels=['Class 0', 'Class 1'],
        title='Test Set Confusion Matrix'
    )
    
    # Feature importance
    feature_importance = model.feature_importances_
    print("\nTop 5 Most Important Features:")
    top_features = np.argsort(feature_importance)[-5:][::-1]
    for idx, feat_idx in enumerate(top_features, 1):
        print(f"{idx}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
