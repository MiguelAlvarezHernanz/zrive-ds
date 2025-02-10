import os
import sys
os.chdir('/home/miguel/zrive-ds')
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from src.module_3.metrics_fun import *

# Load Data
dataset_path = "data/groceries/sampled_box_builder_df.csv"
df = pd.read_csv(dataset_path)

# Filter orders with <5 products
df = df[df.groupby('order_id')['outcome'].transform('sum') >= 5]

# Data processing (categorical encoding and handling time variables) + 3 way split (the function for data processing is called inside the time_based split)
X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(df, date_column='order_date', train_size=0.7, val_size=0.15, test_size=0.15)

# Fit selected model 
model = RidgeClassifier(alpha=1.0)
model.fit(X_train, y_train)

# Determine the optimal threshold to make predictions
# product = int(input('Introduce a Variant_ID: ')) # This should be the interface in production pipeline
product = 34081589887108 # The variant_id for our oat milk is introduced as example

assert product in df['variant_id'].values, f"Error: invalid id or unavailable data for variant_id {product}."
notifications_to_be_sent = 400 # This value has to be adjusted taking into account our company's goals and estimating the cost of sending notifications to uninterested users

prior_sales = compute_last_month_sales(df, product)
precision = 5*prior_sales/notifications_to_be_sent # Ensures an expected result of 25% boost to specific product's sales

optimal_threshold = find_threshold_for_precision(model, X_val, y_val, precision)

# Prediction
decision_scores_val = model.decision_function(X_val)
y_predicted_val = (decision_scores_val >= optimal_threshold).astype(int)

decision_scores = model.decision_function(X_test)
y_predicted = (decision_scores >= optimal_threshold).astype(int)

# Results (sent to the sales/marketing teams)
users_to_be_notified = X_test.loc[y_predicted == 1, 'user_id'].unique()
print(f'Final number of notifications sent: {len(users_to_be_notified)}')

# Evaluation
print('\nEvaluation:')
print(f"Precision needed to fullfil requests: {precision:.3f}")
print(f"Custom Threshold: {optimal_threshold:.3f}")
print("\nClassification Report for Validation Data:\n", classification_report(y_val, y_predicted_val))

print("\nClassification Report for Test Data:\n", classification_report(y_test, y_predicted))