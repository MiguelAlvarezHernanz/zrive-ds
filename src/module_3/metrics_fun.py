import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd

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

def plot_roc_pr_curves(models, X_val, y_val, model_names=None):
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