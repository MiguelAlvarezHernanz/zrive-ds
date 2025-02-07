import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

'''
def plot_roc_pr_curves(model, X_val, y_val):
    """
    Plots the ROC Curve and Precision-Recall Curve for a given model.
    
    Works with models that have either `predict_proba()` (e.g., LogisticRegression)
    or `decision_function()` (e.g., RidgeClassifier, SVM).
    
    Parameters:
    - model: Trained classifier.
    - X_val: Validation features.
    - y_val: True labels for validation data.
    
    Returns:
    - None (Displays the plots).
    """
    # Try using predict_proba, otherwise use decision_function
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val)[:, 1] 
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_val)
    else:
        raise AttributeError("Model must have `predict_proba` or `decision_function`")

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_val, y_scores)
    pr_auc = auc(recall, precision)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    axes[0].plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[0].legend()

    # Precision-Recall Curve
    axes[1].plot(recall, precision, color='green', label=f'PR curve (AUC = {pr_auc:.2f})')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
'''

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
    