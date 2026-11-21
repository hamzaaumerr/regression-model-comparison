
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD REAL DATASET
diabetes = load_diabetes(as_frame=True, scaled=False)

X = diabetes.data
y = diabetes.target

feature_names = X.columns.tolist()

print("========== DATASET INFORMATION ==========")
print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nFeatures:", feature_names)
print("\nFirst 5 Records:")
print(X.head())

# 2. SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. DEFINE MODELS
models = {
    "Linear Regression": LinearRegression(),

    "Ridge (alpha=1)": Ridge(alpha=1.0),

    "Ridge (alpha=10)": Ridge(alpha=10.0),

    "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),

    "Lasso (alpha=1)": Lasso(alpha=1.0, max_iter=10000),

    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000
    )
}

# 4. CROSS-VALIDATION SETUP
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
pipelines = {}

print("\n========== MODEL COMPARISON ==========\n")

print(
    f"{'Model':<25}"
    f"{'Train R2':>12}"
    f"{'CV Mean':>12}"
)

print("-" * 49)

# 5. TRAIN AND EVALUATE MODELS
for name, algorithm in models.items():

    # Create Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", algorithm)
    ])

    # Train using training data
    pipeline.fit(X_train, y_train)

    # Training performance
    train_predictions = pipeline.predict(X_train)

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(
        f"{name:<25}"
        f"{train_r2:>12.4f}"
        f"{cv_mean:>12.4f}"
    )

    # Store results
    results.append({
        "Model": name,
        "Train R2": train_r2,
        "CV Mean": cv_mean,
        "CV Std": cv_std
    })

    pipelines[name] = pipeline

# 6. MODEL COMPARISON
results_df = pd.DataFrame(results)

print("\n========== COMPLETE RESULTS ==========")

print(
    results_df.sort_values(
        by="CV Mean",
        ascending=False
    ).to_string(index=False)
)

# 7. SELECT BEST MODEL USING CV R2
best_result = results_df.loc[
    results_df["CV Mean"].idxmax()
]

best_name = best_result["Model"]
best_model = pipelines[best_name]

print("\n========== BEST MODEL ==========")

print("Best Model:", best_name)
print(f"CV R2: {best_result['CV Mean']:.4f}")

# 8. FINAL TEST EVALUATION
test_predictions = best_model.predict(X_test)

test_r2 = r2_score(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

print("\n========== FINAL TEST EVALUATION ==========")

print(f"Test R2: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# 9. FEATURE IMPORTANCE
print("\n========== FEATURE IMPORTANCE ==========")

best_algorithm = best_model.named_steps["model"]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": best_algorithm.coef_
}).sort_values(
    "Coefficient",
    key=abs,
    ascending=False
)

print(importance.to_string(index=False))

# 10. LASSO FEATURE SELECTION
print("\n========== LASSO FEATURE SELECTION ==========")

lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=0.5, max_iter=10000))
])

lasso_pipeline.fit(X_train, y_train)

lasso_coefficients = (
    lasso_pipeline.named_steps["model"].coef_
)

for name, coef in zip(feature_names, lasso_coefficients):

    if np.isclose(coef, 0.0, atol=1e-8):
        print(f"{name}: {coef:+.4f} (Not Selected)")
    else:
        print(f"{name}: {coef:+.4f} (Important)")

# 11. PREDICT NEW DATA
print("\n========== FINAL PREDICTION ==========")

new_data = pd.DataFrame([{
    "age": 50.0,
    "sex": 1.0,
    "bmi": 25.0,
    "bp": 90.0,
    "s1": 180.0,
    "s2": 110.0,
    "s3": 55.0,
    "s4": 3.5,
    "s5": 4.5,
    "s6": 90.0
}])

prediction = best_model.predict(new_data)

print("New Data:")
print(new_data)

print(f"\nPredicted Target: {prediction[0]:.4f}")