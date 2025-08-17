# Supervised Learning

This project implements supervised learning algorithms for both regression and classification tasks. It provides implementations using perceptrons as well as standard machine learning models like scikit-learn regression and K-Nearest Neighbors (KNN).

## Features

- **Regression**
  - **Perceptron**: Implements a linear regression model using a perceptron.
  - **Scikit-Learn Regression**: Uses `sklearn` for regression with standard preprocessing and evaluation.
  - Evaluates models using Mean Squared Error (MSE) and R² score.
  - Preprocesses student performance data for prediction of grades.

- **Classification**
  - **Perceptron**: Implements a linear classifier for binary classification tasks.
  - **K-Nearest Neighbors (KNN)**: Implements a KNN classifier with optional cross-validation to find the optimal `k`.
  - Evaluates models using classification accuracy.
  - Preprocesses breast cancer diagnostic data for prediction of malignant or benign tumors.

## How to Run

The main entry points are `regression.py` for regression tasks and `classification.py` for classification tasks. You can specify the model via command-line arguments.

### Regression

```bash
# Run regression using Perceptron (default)
python regression.py

# Run regression using scikit-learn model
python regression.py --sklearn
# or shorthand
python regression.py -s
```
### Classification

```bash
# Run classification using Perceptron (default)
python classification.py

# Run classification using K-Nearest Neighbors
python classification.py --knn
# or shorthand
python classification.py -k
```

## Project Structure
```text
.
├── regression.py                       # Runs regression using Perceptron or sklearn
├── classification.py                   # Runs classification using Perceptron or KNN
├── perceptron.py                        # Perceptron implementation for regression/classification
├── knn.py                               # K-Nearest Neighbors implementation
├── sklearn_model.py                     # Wrapper for sklearn regression
├── cross_validate.py                    # Utility to split data into train/test sets
├── preprocess_student_data.py           # Preprocessing for student dataset
├── student-por.csv                       # Dataset for regression tasks
├── student_dataset_description.txt       # Description of the student dataset
├── breast_cancer_diagnostic.csv         # Dataset for classification tasks
├── breast_cancer_diagnostic_description.txt  # Description of the breast cancer dataset
└── README.md                             # Project overview and instructions
