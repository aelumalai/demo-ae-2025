# Module 5 Assignment - ML/AI Berkeley Course
**Student:** Amudha Elumalai

## Overview
This repository contains the implementation for Module 5 of the ML/AI Berkeley course. The assignment focuses on machine learning fundamentals including data preprocessing, model development, training, and evaluation.

## Project Structure
```
demo-ae-2025/
├── data/
│   ├── raw/              # Raw data files (not tracked by git)
│   └── processed/        # Processed data files (not tracked by git)
├── models/               # Saved model files (not tracked by git)
├── src/                  # Source code for utilities
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── model_utils.py
├── module5_assignment.ipynb  # Main assignment notebook
├── train.py              # Training script
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/aelumalai/demo-ae-2025.git
cd demo-ae-2025
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook module5_assignment.ipynb
```

## Usage

### Working with the Notebook
The main assignment is implemented in `module5_assignment.ipynb`. Open it in Jupyter Notebook or JupyterLab and follow the sections:

1. **Setup and Imports** - Import required libraries
2. **Data Loading and Exploration** - Load and explore the dataset
3. **Data Preprocessing** - Clean and prepare data
4. **Exploratory Data Analysis** - Visualize data patterns
5. **Model Development** - Build and train ML models
6. **Model Evaluation** - Evaluate model performance
7. **Results and Conclusions** - Summarize findings

### Using Utility Functions
The `src/` directory contains helper functions for common ML tasks:

```python
from src import load_data, scale_features, evaluate_model

# Load data
data = load_data('data/raw/dataset.csv')

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Evaluate model
metrics = evaluate_model(y_true, y_pred, model_name='My Model')
```

### Training a Model
Use the provided training script:
```bash
python train.py
```

## Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tensorflow >= 2.10.0
- jupyter >= 1.0.0

See `requirements.txt` for complete list.

## Assignment Tasks
- [ ] Load and explore the dataset
- [ ] Perform data preprocessing and cleaning
- [ ] Conduct exploratory data analysis
- [ ] Develop machine learning model(s)
- [ ] Train and validate models
- [ ] Evaluate model performance
- [ ] Document findings and conclusions

## Notes
- Make sure to place your data files in the `data/raw/` directory
- Processed data will be saved in `data/processed/`
- Trained models will be saved in `models/`
- All data and model files are excluded from git tracking

## License
This is an academic assignment for the ML/AI Berkeley Course.

## Contact
For questions or issues, please contact the course instructors.
