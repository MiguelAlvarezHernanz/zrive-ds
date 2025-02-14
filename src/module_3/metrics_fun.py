import os
import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(df, time_columns=['created_at', 'order_date'], freq_encode_cols=['product_type', 'vendor']):
    """
    Processes time-related features and applies frequency encoding to categorical variables.

    Parameters:
    - df (pd.DataFrame): The dataset containing time and categorical columns.
    - time_columns (list): List of columns containing time-related data.
    - freq_encode_cols (list): List of categorical columns to apply frequency encoding.

    Returns:
    - df_processed (pd.DataFrame): The transformed dataset.
    """
    df_processed = df.copy()  # Work on a copy to avoid modifying original data

    # Handle time-related features
    df_processed['hour'] = pd.to_datetime(df_processed[time_columns[0]]).dt.hour
    df_processed['day_of_week'] = pd.to_datetime(df_processed[time_columns[1]]).dt.dayofweek
    
    # Drop original time columns
    df_processed = df_processed.drop(columns=time_columns)

    # Apply Frequency Encoding for categorical variables
    for col in freq_encode_cols:
        freq_map = df_processed[col].value_counts(normalize=True)  # Compute frequency
        df_processed[col] = df_processed[col].map(freq_map)

    return df_processed


def sequential_train_test_split(df, test_size=0.15, random_state=42):
    """
    df: pandas Data.Frame with a 'user_id' feature (needs to be sorted by that feature).
    test_size: float between 0 and 1 (predetermined to 0.15 in order to get a 70/15/15 split).
    random_state: an integer seed for the random split (predetermined to 42).

    Ensures that if an order with user_order_seq = n is in train, then all previous ones (1, ..., n-1) are also in
    train by splitting unique users into train & test.
    """
    
    unique_users = df['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)

    # Keep all rows of a user in the assigned split
    df_train = df[df['user_id'].isin(train_users)]
    df_test = df[df['user_id'].isin(test_users)]

    return df_train, df_test


def time_based_split(df, date_column='order_date', train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Performs a time-wise split of the dataset based on chronological order.

    Parameters:
    - df (pd.DataFrame): The dataset containing a datetime column and an 'outcome' column.
    - date_column (str): The name of the column with datetime values.
    - train_size (float): Proportion of data for training (default = 70%).  |
    - val_size (float): Proportion of data for validation (default = 20%).  |> Those three must sum to 1.
    - test_size (float): Proportion of data for testing (default = 10%).    |

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    # Compute split indices
    train_idx = int(len(df) * train_size)
    val_idx = train_idx + int(len(df) * val_size)

    # Split dataset
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]

    # Preprocess dfs
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    test_df = preprocess_data(test_df)

    # Define X (features) and y (target)
    X_train, y_train = train_df.drop(columns=['outcome', 'variant_id', 'user_id', 'order_id']), train_df['outcome']
    X_val, y_val = val_df.drop(columns=['outcome', 'variant_id', 'user_id', 'order_id']), val_df['outcome']
    X_test, y_test = test_df.drop(columns=['outcome', 'variant_id', 'user_id', 'order_id']), test_df['outcome']

    return X_train, X_val, X_test, y_train, y_val, y_test


'''def plot_roc_pr_curves(models, X_val, y_val, model_names=None):
    """
    Plots the ROC Curve and Precision-Recall Curve for one or multiple models.
    
    Works with models that have either `predict_proba()` (e.g., LogisticRegression)
    or `decision_function()` (e.g., RidgeClassifier, SVM).
    
    Parameters:
    - models: A single model or a list of trained classifiers.
    - X_val: Validation features.
    - y_val: True labels for validation data.
    - model_names: Optional list of model names (must match the length of `models`).
    
    Returns:
    - None (Displays the plots).
    """
    # Ensure models is a list
    if not isinstance(models, list):
        models = [models] 

    # Assign default names if none are provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Iterate over models
    for model, name in zip(models, model_names):
        # Try using predict_proba, otherwise use decision_function
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_val)[:, 1] 
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_val)
        else:
            raise AttributeError(f"Model {name} must have `predict_proba` or `decision_function`")

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(y_val, y_scores)
        pr_auc = auc(recall, precision)

        # Plot ROC Curve
        axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # Plot Precision-Recall Curve
        axes[1].plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

    # Format ROC plot
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[0].legend()

    # Format Precision-Recall plot
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()''' # Outdated version of function


def plot_roc_pr_curves(models, X_vals, y_val, model_names=None):
    """
    Plots the ROC Curve and Precision-Recall Curve for one or multiple models.
    
    Works with models that have either `predict_proba()` (e.g., LogisticRegression)
    or `decision_function()` (e.g., RidgeClassifier, SVM).

    Parameters:
    - models: A single model or a list of trained classifiers.
    - X_vals: A single validation feature set or a list of feature sets (one per model).
    - y_val: True labels for validation data.
    - model_names: Optional list of model names (must match the length of `models`).

    Returns:
    - None (Displays the plots).
    """
    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]
    if not isinstance(X_vals, list):
        X_vals = [X_vals]

    if len(X_vals) == 1:
        X_vals = [X_vals[0].copy() for _ in range(len(models))]

    # Ensure the number of models matches the number of feature sets
    assert len(models) == len(X_vals), "Number of models must match the number of feature sets provided."

    # Assign default names if none are provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Iterate over models and their respective feature sets
    for model, X_val, name in zip(models, X_vals, model_names):
        # Try using predict_proba, otherwise use decision_function
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_val)
        else:
            raise AttributeError(f"Model {name} must have `predict_proba` or `decision_function`")

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(y_val, y_scores)
        pr_auc = auc(recall, precision)

        # Plot ROC Curve
        axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # Plot Precision-Recall Curve
        axes[1].plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

    # Format ROC plot
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[0].legend()

    # Format Precision-Recall plot
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def compute_last_month_sales(df, variant_id):
    """
    Computes the number of sales (count of 'outcome') for a given variant_id 
    during the last available month in the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset containing 'variant_id', 'outcome', and 'order_date'.
    - variant_id (int): The variant_id to filter.

    Returns:
    - int: Number of sales for the given variant_id in the last available month.
    """
    # Ensure 'order_date' is in datetime format
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Find the latest available month in the dataset
    last_month = df['order_date'].max().to_period('M')

    # Filter data to last month only
    last_month_df = df[df['order_date'].dt.to_period('M') == last_month]

    # Filter for the given variant_id and count 'outcome' occurrences
    sales_count = last_month_df[(last_month_df['variant_id'] == variant_id) & (last_month_df['outcome'] == 1)].shape[0]

    return sales_count

def find_threshold_for_precision(model, X_val, y_val, target_precision):
    """
    Finds the decision threshold for RidgeClassifier that results in a given precision.

    Parameters:
    - model: Trained RidgeClassifier model.
    - X_val: Validation set features.
    - y_val: True labels.
    - target_precision: The precision level we want to achieve.

    Returns:
    - optimal_threshold: The threshold value that leads to the target precision.
    """
    # Get decision scores instead of probabilities
    decision_scores = model.decision_function(X_val)

    # Compute precision-recall pairs for different thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, decision_scores)

    # Find the closest precision to the target
    threshold_index = np.argmin(np.abs(precisions - target_precision))

    # Get the corresponding threshold
    optimal_threshold = thresholds[threshold_index]

    return optimal_threshold


def save_model(model, model_name: str, output_path='models'):
    os.chdir('/home/miguel/zrive-ds')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    joblib.dump(model, os.path.join(output_path, model_name))