# **Milestone 1: exploration phase**
### **1- Load dataset & filter** (restrict to orders of at least 5 items)


```python
import pandas as pd
import os
import numpy as np

os.chdir('/home/miguel/zrive-ds')

dataset_path = "data/groceries/sampled_box_builder_df.csv"
df = pd.read_csv(dataset_path)
raw_length = df.shape[0]
df = df[df.groupby('order_id')['outcome'].transform('sum') >= 5]
filtered_length = df.shape[0]
print(f'Number of rows reduced by {round(filtered_length/raw_length*100, 2)}%, from {raw_length} raw, to {filtered_length} filtered.')
```

    Number of rows reduced by 75.12%, from 2880549 raw, to 2163953 filtered.


### **2- Categorical encoding & data processing**


```python
# Handle time-related features
df['hour'] = pd.to_datetime(df['created_at']).dt.hour
df['day_of_week'] = pd.to_datetime(df['order_date']).dt.dayofweek # Already cardinally encoded
df_filtered = df.drop(columns=['created_at', 'order_date'])

# Frequency Encoding for 'product_type' and 'vendor'
for col in ['product_type', 'vendor']:
    freq_map = df_filtered[col].value_counts(normalize=True)  # Compute frequency
    df_filtered[col] = df_filtered[col].map(freq_map)

df_filtered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>...</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
      <th>hour</th>
      <th>day_of_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>0.044619</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>0.044619</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>0.044619</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>0.044619</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>0.044619</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



In first place, it is important to be aware of the fact that linear models do not handle raw datetime values well, and in their raw form, they do not provide useful numerical information. However, I considered that it would be interesting to keep some information from those features. There might be any correlation between the hour in which the push notification is sent and the outcome chance. Same for the week day: it may be more efficient to push users on Fridays than on Wednesdays.

Facing the categorical feature encoding, it is absolutely unefficient to perform one-hot encoding, as it leaves a dataframe with 349 columns (curse of dimensionality). Frecuency encoding is proposed over it.

Finally, I would like to highlight the possibility of performing Target Encoding (mean Outcome encoding), this can be useful if the information leakage is correctly avoided and will be tried later if possible.

### **3- Three-way split**
**We will do it trying to ensure Sequential Integrity (avoid information leakage through `user_order_seq`)**


```python
from sklearn.model_selection import train_test_split

# Ensure sequential integrity
df_sorted = df_filtered.sort_values(by=['user_id', 'user_order_seq'])

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

# Split into train, validation, and test sets -> 70/20/10
df_train, df_temp = sequential_train_test_split(df_sorted, 0.2)
df_val, df_test = sequential_train_test_split(df_temp, 0.1)

# Define target and features (drop ids as well)
X_train = df_train.drop(columns=['outcome', 'variant_id', 'user_id', 'order_id'])  # Features
y_train = df_train['outcome']  # Target

X_val = df_val.drop(columns=['outcome', 'variant_id', 'user_id', 'order_id'])  # Features
y_val = df_val['outcome']  # Target

X_test = df_test.drop(columns=['outcome', 'variant_id', 'user_id', 'order_id'])  # Features
y_test = df_test['outcome']  # Target
```

Here, a new function is introduced in order to perform the 3-way split over unique users, instead of doing it over rows, with the aim of avoiding information leakage. I know that, in production, past order information will be available for clients who have already ordered, so the optimal way to split data may be to do it time-wise. That will be tried later. This approach might be good in order to predict the behaviour of new clients. (I leave here noted the posibility of implementing 2 models: one for users who are already clients, and another for new customers).

### **3- Definition of evaluation metrics**

We will be using ROC and Precision-Recall curves as main evaluation metrics for our models. It will be important to be aware of the fact that a diagonal line is the baseline for ROC curve: the expected result when predictions are randomly made. However, this is not always true for PR curve (it is not in our case), since there is a high imbalance within our target feature `outcome`.

Therefore, the baseline precision (the precision expected for random predictions) in our case will be computed as follows:


```python
baseline_precision = df['outcome'].sum() / len(df)
print(f"Baseline Random Precision: {baseline_precision:.4f}")
```

    Baseline Random Precision: 0.0145


### **4- Modeling for unique user split**
**First Model** (Logistic regression with L2 regularisation)


```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ridge_logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(penalty='l2', C=1e-4))
])

ridge_logistic_pipeline.fit(X_train, y_train)

from src.module_3.metrics_fun import plot_roc_pr_curves
plot_roc_pr_curves(ridge_logistic_pipeline, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_12_0.png)
    



```python
lasso_logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(penalty='l1', solver='saga', C=1e-4))
])

lasso_logistic_pipeline.fit(X_train, y_train)

plot_roc_pr_curves(lasso_logistic_pipeline, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_13_0.png)
    



```python
from sklearn.linear_model import RidgeClassifier

ridge_model = RidgeClassifier(alpha=1.0)
ridge_model.fit(X_train, y_train)

plot_roc_pr_curves(ridge_model, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_14_0.png)
    



```python
ridge_model_weighted = RidgeClassifier(class_weight="balanced", alpha=1.0)
ridge_model_weighted.fit(X_train, y_train)

plot_roc_pr_curves(ridge_model_weighted, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_15_0.png)
    



```python
plot_roc_pr_curves([ridge_logistic_pipeline, lasso_logistic_pipeline, ridge_model, ridge_model_weighted], X_val, y_val, model_names=['Ridge Logistic', 'Lasso Logistic', 'Ridge Classifier', 'Ridge Classifier weighted'])
```


    
![png](exploration_phase_files/exploration_phase_16_0.png)
    


We can see here that the Ridge Linear Classifier is the best performing model overall, but it will be important to take into account that its weighted version works better when high precissions (over ~0.7) are needed. There is another region (between 0.3-0.4 precission) where the logistic models (both Lasso & Ridge) perform better as well.

Taking into account how similar the results obtained with Lasso & Ridge logistic regresions are, we can assume that regularisation does not make a big difference in this problem. Even though that, I tryed doing some 'grid search', implementing different coefficients for both Lasso and ridge regularisations. These models are not included since, as we assumed, changes in regularization do not affect the models' performance in much relevant ways.

However, we will see later the value of Lasso regulation as a feature selector.

### **5- Business insights**
**It is important to define our assumptions and objectives in order to determine a threshold to make predictions.**

- We assume that our push notifications will follow the same current open rate in our app, around 5%.
- We assume that, if a positive prediction made by our model is correct, and the user opens the push notification, he will buy the product. This here, means that we can understand the `precision` of our model as the probability of bought, conditioned to the probability of the user opening his notification.

We consider the Probability of Success (PoS) as the global probability of bought for a push notification to a determined user, and the Number of Successes (NoS) as the expected number of sales achieved through the push notification system.

- $PoS = \frac{5}{100} \cdot precision$
- $NoS = \frac{5}{100} \cdot precision \cdot N$ , where N is the number of notifications sent.

As we want to achieve a 25% boost over the sales of a determined product, the precision needed (and therefore the threshold applied to our model) can be computed as follows:

$Precision = \frac{100}{4 \cdot 5} \cdot \frac{V}{N} = 5 \frac{V}{N}$ , where V is the total sales for that product over the last month (the number that will be increased by a 25%)

Example:


```python
product = 33826472919172 # A given variant_id for the example
product2 = 34081589887108 # Oat milk (selected top product to avoid products sold once)

from src.module_3.metrics_fun import compute_last_month_sales
V = compute_last_month_sales(df, product2)

import matplotlib.pyplot as plt
import numpy as np

precision_needed = lambda N: 5*V/N
N_list = np.arange(200, 2000, 100)

plt.plot(N_list, precision_needed(N_list), marker='o', linestyle='-')
plt.xlabel("Number of notifications")
plt.ylabel("Precision Needed")
plt.grid(True)
plt.show()
```


    
![png](exploration_phase_files/exploration_phase_19_0.png)
    


Visualizing this curve is important because it gives us an idea of the number of 'uninterested users' that we will notify. For example: any point of this curve is expected to produce the same outcome (concerning sales) but we can achieve that outcome by notifying 175 users with a perfect precision or by notifying 1750 users with a precision of 0.1.

The limitation here is that we cannot simply choose the point with the highest possible precision. This is explained by the Precision-Recall curve: A high precision usually implies a low recall, which means that we are 'finding' a low percentage of the actual number of potential buyers (even though that we are very confident that the ones we found are going to buy; high precision). When that number of real potential buyers is not very high (as it is in this case) it might not be possible to achieve our goals by using a very high precision model: we will have to take higher risks and lower the precision (but always acknowledging the cost of sending ignored notifications).

After choosing the precision that we want to work with, we will need to find the threshold that will have to be used in our model in order to achieve the specified precision. 

### **6- Changing our data split: time wise splitting**


```python
from src.module_3.metrics_fun import time_based_split

X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(df, date_column='order_date', train_size=0.7, val_size=0.2, test_size=0.1)
```

### **7- Modeling for time-based split**
**First Model** (Logistic regression with L2 regularisation)


```python
ridge_logistic_pipeline_T = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(penalty='l2', C=1e-4))
])

ridge_logistic_pipeline_T.fit(X_train, y_train)

from src.module_3.metrics_fun import plot_roc_pr_curves
plot_roc_pr_curves(ridge_logistic_pipeline_T, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_24_0.png)
    



```python
lasso_logistic_pipeline_T = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(penalty='l1', solver='saga', C=1e-4))
])

lasso_logistic_pipeline_T.fit(X_train, y_train)

plot_roc_pr_curves(lasso_logistic_pipeline_T, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_25_0.png)
    



```python
ridge_model_T = RidgeClassifier(alpha=1.0)
ridge_model_T.fit(X_train, y_train)

plot_roc_pr_curves(ridge_model_T, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_26_0.png)
    



```python
ridge_model_weighted_T = RidgeClassifier(class_weight="balanced", alpha=1.0)
ridge_model_weighted_T.fit(X_train, y_train)

plot_roc_pr_curves(ridge_model_weighted_T, X_val, y_val)
```


    
![png](exploration_phase_files/exploration_phase_27_0.png)
    



```python
plot_roc_pr_curves([ridge_logistic_pipeline_T, lasso_logistic_pipeline_T, ridge_model_T, ridge_model_weighted_T], X_val, y_val, model_names=['Ridge Logistic', 'Lasso Logistic', 'Ridge Classifier', 'Ridge Classifier weighted'])
```


    
![png](exploration_phase_files/exploration_phase_28_0.png)
    


Here, we can see a slightly better performance than what we obtained splitting our data by unique users. This makes sense since some users appear in both train and test, and their behaviour will be better-related. Furthermore, this approach (time split) is even more resistant to information leakage and its train information is more similar to what we will have available at production.

Now, let's dive into evaluating the relevance/impact of each feature in our models:


```python
ridge_coefs = pd.DataFrame(
    {
        "features": X_train.columns.tolist(),
        "importance": np.abs(ridge_logistic_pipeline_T.named_steps['model'].coef_[0]),
        "regularisation": ['l2'] * len(X_train.columns.tolist()),
    }
)
ridge_coefs = ridge_coefs.sort_values("importance", ascending=True)


lasso_coefs = pd.DataFrame(
    {
        "features": X_train.columns.tolist(),
        "importance": np.abs(lasso_logistic_pipeline_T.named_steps['model'].coef_[0]),
        "regularisation": ['l1'] * len(X_train.columns.tolist()),
    }
)
lasso_coefs = lasso_coefs.sort_values("importance", ascending=True)


coefs_df = pd.concat([ridge_coefs, lasso_coefs])

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

for regularisation, color in zip(["l2", "l1"], ["blue", "orange"]):
    subset = coefs_df[coefs_df["regularisation"] == regularisation]
    ax.barh(subset["features"], subset["importance"], alpha=0.7, label=f"{regularisation}", color=color)

ax.set_xlabel("Absolute Coefficient Value")
ax.set_ylabel("Features")
ax.set_title("Comparison of Feature Importance: Ridge vs. Lasso")
# ax.set_xlim(0, 1)
ax.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
```


    
![png](exploration_phase_files/exploration_phase_31_0.png)
    


Here, we can see how Lasso regularisation actually acts as promised, returning coefficients equal to 0 to every feature except of those three: `ordered_before`, `global_popularity` y `abandoned_before`. Since the metrics obtained for Lasso regression are similar to the rest of the models implemented, we will choose to model using only these 3 features, just by simplicity:


```python
ridge_pipeline_reduced = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(penalty='l1', solver='saga', C=1e-4))
])

ridge_pipeline_reduced.fit(X_train[['ordered_before', 'global_popularity', 'abandoned_before']], y_train)

plot_roc_pr_curves([ridge_logistic_pipeline, ridge_pipeline_reduced], [X_val, X_val[['ordered_before', 'global_popularity', 'abandoned_before']]], y_val, model_names=['Using all features', 'Using 3 features'])
```


    
![png](exploration_phase_files/exploration_phase_33_0.png)
    


Here we see that the result obtained is pretty similar to what we obtained using all features.
