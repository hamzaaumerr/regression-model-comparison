# Regression Model Comparison Using Linear, Ridge, Lasso & ElasticNet

A Machine Learning project that compares multiple regression algorithms using a real-world dataset from Scikit-learn. The project explores regularization, cross-validation, model selection, coefficient analysis, and feature selection.

## Project Overview

This project compares six regression configurations using the Scikit-learn Diabetes Dataset.

The objective is to understand how Linear Regression and regularization techniques such as Ridge, Lasso, and ElasticNet perform on real-world data.

The project also demonstrates how to build Scikit-learn Pipelines, prevent preprocessing data leakage, select the best-performing model using cross-validation, and evaluate its performance on unseen test data.

## Dataset

- **Dataset:** Diabetes Dataset
- **Source:** Scikit-learn
- **Samples:** 442
- **Input Features:** 10
- **Target:** Quantitative measure of disease progression one year after baseline
- **Data Type:** Real-world numerical research data

The dataset is loaded directly using `sklearn.datasets.load_diabetes()`. No manual CSV download is required.

### Dataset Features

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex coding in the dataset |
| bmi | Body Mass Index |
| bp | Average blood pressure |
| s1 | Total serum cholesterol |
| s2 | LDL cholesterol |
| s3 | HDL cholesterol |
| s4 | Total cholesterol / HDL ratio |
| s5 | Serum measurement associated with triglycerides |
| s6 | Blood sugar measurement |

**Dataset References:**

- [Scikit-learn Diabetes Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Original Diabetes Dataset Source](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)

## Technologies & Libraries

- **Python** – Programming language
- **NumPy** – Numerical operations
- **Pandas** – Dataset handling and result comparison
- **Scikit-learn** – Regression models, preprocessing, Pipelines, cross-validation and evaluation

## Models Implemented

The project compares six regression configurations:

| Model | Configuration |
|---|---|
| Linear Regression | Standard Linear Regression |
| Ridge Regression | Alpha = 1.0 |
| Ridge Regression | Alpha = 10.0 |
| Lasso Regression | Alpha = 0.1 |
| Lasso Regression | Alpha = 1.0 |
| ElasticNet | Alpha = 0.1, L1 Ratio = 0.5 |

## Machine Learning Workflow

### 1. Dataset Loading

- Load the real Diabetes Dataset using Scikit-learn.
- Retrieve 442 observations and 10 input features.
- Use original feature values with `scaled=False`.
- Separate the input features and target variable.

### 2. Train-Test Split

Split the dataset into:

- 80% training data
- 20% testing data

Use `random_state=42` for reproducibility.

### 3. Feature Scaling

Apply StandardScaler to standardize input features.

Feature scaling is included inside a Scikit-learn Pipeline to prevent preprocessing data leakage.

### 4. Model Training

Train and compare six regression configurations using:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

### 5. Cross-Validation

Apply shuffled 5-fold cross-validation using KFold.

Each validation fold independently fits its own StandardScaler and regression model.

### 6. Model Selection

Select the best model based on the highest mean cross-validation R² score.

### 7. Final Model Evaluation

Evaluate the selected model on the previously unseen test dataset using:

- Test R² Score
- Root Mean Squared Error (RMSE)

### 8. Feature Importance

Analyze regression coefficients to understand the relative contribution and direction of each standardized feature.

### 9. Lasso Feature Selection

Apply Lasso Regression with alpha = 0.5 to identify features whose coefficients are reduced to zero.

### 10. New Data Prediction

Use the selected model to predict the target value for a new example containing all 10 input features.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Train R² Score | Measures model fit on training data |
| Cross-Validation R² | Measures performance across multiple validation folds |
| Test R² Score | Measures predictive performance on unseen data |
| RMSE | Measures prediction error in target units |

The best model is selected using cross-validation rather than test performance.

## Project Structure

```text
regression-model-comparison/
│
├── regression_model_comparison.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hamzaaumerr/regression-model-comparison.git
cd regression-model-comparison
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

```bash
python regression_model_comparison.py
```