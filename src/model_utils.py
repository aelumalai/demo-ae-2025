"""
Model utilities for Module 5 Assignment.

This module contains helper functions for model training,
evaluation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list): List of label names
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy']):
    """
    Plot training history for deep learning models.
    
    Args:
        history: Training history object from model.fit()
        metrics (list): List of metrics to plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history.history:
            axes[i].plot(history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history.history:
                axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} vs Epoch')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print detailed classification report.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        target_names (list): List of target class names
    """
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
