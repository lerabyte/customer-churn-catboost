from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
DECISION_THRESHOLD = 0.50

DATA_FILE = "Telco-Customer-Churn.csv"


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def prepare_features(
    dataframe,
    categorical_columns,
    numerical_columns,
    training_medians,
):
    """
    Fill missing values using information learned from
    the training data only.
    """

    dataframe = dataframe.copy()

    dataframe[categorical_columns] = (
        dataframe[categorical_columns]
        .fillna("Missing")
        .astype(str)
    )

    dataframe[numerical_columns] = (
        dataframe[numerical_columns]
        .fillna(training_medians)
    )

    return dataframe


def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate model metrics at a selected probability threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_churners": int(tp + fn),
    }


def compare_thresholds(y_true, probabilities, thresholds):
    """
    Compare precision, recall, and errors at different thresholds.
    """

    results = []

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        results.append(
            {
                "Threshold": threshold,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "False Positives": metrics["false_positives"],
                "False Negatives": metrics["false_negatives"],
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 3. LOAD THE DATA
# ---------------------------------------------------------

data_path = Path(DATA_FILE)

if not data_path.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. "
        "Place the CSV in the same folder as this script."
    )

df = pd.read_csv(data_path)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 4. EXPLORE THE TARGET
# ---------------------------------------------------------

print("\nChurn counts:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

df["Churn"].value_counts().reindex(
    ["No", "Yes"]
).plot(
    kind="bar",
    title="Customer Churn Distribution",
)

plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 5. CLEAN THE DATA
# ---------------------------------------------------------

# TotalCharges may load as text because some rows contain blanks.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove the customer ID because it is only an identifier.
columns_to_drop = ["customerID"]

# These extra columns are included as safety checks in case
# a different version of the dataset is used.
possible_leakage_columns = [
    "Churn Reason",
    "ChurnReason",
    "Cancellation Date",
    "Customer Status",
]

for column in possible_leakage_columns:
    if column in df.columns:
        columns_to_drop.append(column)

df = df.drop(
    columns=columns_to_drop,
    errors="ignore",
)

# Convert the target from Yes/No into 1/0.
df["Churn"] = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1,
    }
)

if df["Churn"].isna().any():
    raise ValueError(
        "The Churn column contains unexpected values."
    )

df["Churn"] = df["Churn"].astype(int)


# ---------------------------------------------------------
# 6. DEFINE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(columns="Churn")
y = df["Churn"]

print("\nNumber of features:")
print(X.shape[1])


# ---------------------------------------------------------
# 7. CREATE TRAIN, VALIDATION, AND TEST SETS
# ---------------------------------------------------------

# First, create the untouched test set.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Then create a validation set from the training data.
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VALIDATION_SIZE,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)

print("\nData split sizes:")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)

print("\nChurn rates:")
print("Training:", round(y_train.mean(), 4))
print("Validation:", round(y_validation.mean(), 4))
print("Testing:", round(y_test.mean(), 4))


# ---------------------------------------------------------
# 8. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ---------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ---------------------------------------------------------
# 9. HANDLE MISSING VALUES
# ---------------------------------------------------------

# Learn the median values from the training data only.
training_medians = X_train[numerical_columns].median()

X_train = prepare_features(
    X_train,
    categorical_columns,
    numerical_columns,
    training_medians,
)

X_validation = prepare_features(
    X_validation,
    categorical_columns,
    numerical_columns,
    training_medians,
)

X_test = prepare_features(
    X_test,
    categorical_columns,
    numerical_columns,
    training_medians,
)


# ---------------------------------------------------------
# 10. BUILD THE CATBOOST MODEL
# ---------------------------------------------------------

model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="PRAUC:type=Classic;use_weights=False",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    verbose=100,
    allow_writing_files=False,
)


# ---------------------------------------------------------
# 11. TRAIN THE MODEL
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train,
    cat_features=categorical_columns,
    eval_set=(X_validation, y_validation),
    early_stopping_rounds=100,
    use_best_model=True,
)

print("\nBest iteration:")
print(model.get_best_iteration())


# ---------------------------------------------------------
# 12. COMPARE DIFFERENT THRESHOLDS
# ---------------------------------------------------------

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

threshold_results = compare_thresholds(
    y_validation,
    validation_probabilities,
    thresholds=[
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ],
)

print("\nValidation threshold comparison:")
print(
    threshold_results
    .round(3)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 13. EVALUATE ON THE TEST SET
# ---------------------------------------------------------

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= DECISION_THRESHOLD
).astype(int)

metrics = calculate_metrics(
    y_test,
    test_probabilities,
    DECISION_THRESHOLD,
)

print("\nTEST RESULTS")
print("-" * 50)

print(
    f"Decision threshold: {metrics['threshold']:.2f}"
)

print(
    f"PR-AUC: {metrics['pr_auc']:.4f}"
)

print(
    f"ROC-AUC: {metrics['roc_auc']:.4f}"
)

print(
    f"Precision for churn: {metrics['precision']:.4f}"
)

print(
    f"Recall for churn: {metrics['recall']:.4f}"
)

print(
    f"F1 score for churn: {metrics['f1']:.4f}"
)

print(
    f"Total actual churners: {metrics['total_churners']}"
)

print(
    f"Correctly caught churners: "
    f"{metrics['true_positives']}"
)

print(
    f"Missed churners: "
    f"{metrics['false_negatives']}"
)

print(
    f"False alarms: "
    f"{metrics['false_positives']}"
)

print(
    f"Correctly predicted customers who stayed: "
    f"{metrics['true_negatives']}"
)


# ---------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Stayed",
            "Churned",
        ],
        digits=4,
    )
)


# ---------------------------------------------------------
# 15. CONFUSION MATRIX
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=[
        "Stayed",
        "Churned",
    ],
    values_format="d",
)

plt.title(
    f"Confusion Matrix at Threshold "
    f"{DECISION_THRESHOLD:.2f}"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 16. PRECISION-RECALL CURVE
# ---------------------------------------------------------

PrecisionRecallDisplay.from_predictions(
    y_test,
    test_probabilities,
    name="CatBoost",
)

plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 17. TEST THE MODEL ON A NEW CUSTOMER
# ---------------------------------------------------------

new_customer = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.00,
            "TotalCharges": 190.00,
        }
    ]
)

# Make sure the new customer has the exact same
# columns and order as the training data.
new_customer = new_customer.reindex(
    columns=X_train.columns
)

new_customer = prepare_features(
    new_customer,
    categorical_columns,
    numerical_columns,
    training_medians,
)

new_customer_probability = model.predict_proba(
    new_customer
)[0, 1]

if new_customer_probability >= DECISION_THRESHOLD:
    new_customer_prediction = "LIKELY TO CHURN"
else:
    new_customer_prediction = "LIKELY TO STAY"

print("\nNEW CUSTOMER PREDICTION")
print("-" * 50)

print(
    f"Churn probability: "
    f"{new_customer_probability:.1%}"
)

print(
    f"Prediction: "
    f"{new_customer_prediction}"
)


# ---------------------------------------------------------
# 18. FEATURE IMPORTANCE
# ---------------------------------------------------------

feature_importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)

print("\nTop 10 most important features:")
print(
    feature_importance
    .head(10)
    .round(3)
)

feature_importance.head(10).sort_values().plot(
    kind="barh",
    title="Top 10 CatBoost Feature Importances",
)

plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 19. SAVE THE MODEL
# ---------------------------------------------------------

model.save_model(
    "catboost_churn_model.cbm"
)

print(
    "\nModel saved as "
    "'catboost_churn_model.cbm'"
)
