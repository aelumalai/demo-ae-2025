"""
Module 5 Assignment - ML/AI Berkeley Course
Source code package initialization.
"""

from .data_preprocessing import (
    load_data,
    handle_missing_values,
    scale_features,
    encode_categorical
)

from .model_utils import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report
)

__all__ = [
    'load_data',
    'handle_missing_values',
    'scale_features',
    'encode_categorical',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_training_history',
    'print_classification_report'
]
