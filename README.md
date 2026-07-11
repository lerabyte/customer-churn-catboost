# Customer Churn Prediction with CatBoost

This project uses machine learning to predict whether a telecom customer is likely to leave the company.

The model is trained on customer account information such as contract type, tenure, internet service, payment method, monthly charges, and technical support. Instead of only producing a yes-or-no prediction, the model returns a churn probability that can be used to rank customers by risk.

## Model

The project uses a CatBoost classifier, a gradient-boosted decision-tree model that works well with structured datasets containing both numerical and categorical features.

Each new decision tree attempts to correct errors made by the previous trees, allowing the final model to learn more complex customer patterns.

## Dataset

The project uses the Telco Customer Churn dataset.

Each row represents one customer, and the target column is:

- `Churn = 1`: Customer left
- `Churn = 0`: Customer stayed

Example features include:

- Contract type
- Customer tenure
- Internet service
- Monthly charges
- Total charges
- Payment method
- Technical support
- Online security

## Data Preparation

The project performs the following preprocessing steps:

- Drops the `customerID` column
- Converts `TotalCharges` into a numerical feature
- Converts the Churn target from Yes/No into 1/0
- Separates categorical and numerical features
- Splits the data into training, validation, and testing sets
- Preserves the churn ratio using stratified splitting

## Model Settings

The main CatBoost settings include:

```python
model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    auto_class_weights="Balanced",
    random_seed=42
)
````

* `iterations`: Maximum number of decision trees
* `depth`: Complexity of each tree
* `learning_rate`: Size of each training adjustment
* `auto_class_weights`: Gives more attention to the smaller churn class

## Results

The model produced the following test results:

| Metric    | Result |
| --------- | -----: |
| PR-AUC    | 0.6645 |
| ROC-AUC   | 0.8438 |
| Precision | 0.5106 |
| Recall    | 0.7727 |
| F1 Score  | 0.6149 |

At a 50% decision threshold, the model:

* Correctly identified 289 churners
* Missed 85 churners
* Produced 277 false alarms
* Correctly identified 758 customers who stayed

## Understanding the Metrics

**PR-AUC:** Measures how well the model ranks churners above customers who stay across different thresholds.

**Precision:** When the model flags a customer as likely to churn, it is correct about 51% of the time.

**Recall:** The model catches about 77% of customers who actually churn.

## Decision Thresholds

The model outputs a probability rather than a guaranteed prediction.

* A 25% threshold flags more customers and catches more churners, but creates more false alarms.
* A 50% threshold provides a middle ground between precision and recall.
* A 75% threshold only flags the highest-risk customers, but misses more churners.

## Example Prediction

A new customer with a month-to-month contract, fiber-optic internet, no technical support, electronic-check payments, and only two months of tenure received:

```text
Churn probability: 90.6%
Prediction: Likely to churn
```

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn catboost
```

## Run the Project

Place the dataset in the project folder and run:

```bash
python customer_churn_catboost.py
```
