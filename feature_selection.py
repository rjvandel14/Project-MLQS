import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Chapter7.FeatureSelection import FeatureSelectionClassification


df_train = pd.read_csv("all_features_train.csv",parse_dates=["Timestamp"])
df_test = pd.read_csv("all_features_test.csv",parse_dates=["Timestamp"])

# NaN values that are now present are because of de devision by 0
# So we can fill them in with 0
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

# Define features and target
X_train = df_train.drop(columns=['target_majority_category'])
X_train = X_train.select_dtypes(include=[np.number])
y_train = df_train['target_majority_category']
X_test = df_test.drop(columns=['target_majority_category'])
X_test = X_test.select_dtypes(include=[np.number])
y_test = df_test['target_majority_category']

# Initialize the feature selection class
fs = FeatureSelectionClassification()

# Perform forward selection
selected_forward, ordered_forward, scores_forward = fs.forward_selection(
    max_features=8,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    algorithm = "random_forest",
    gridsearch=False
)

print("Forward selected features:", selected_forward)
print("Forward scores:", scores_forward)
# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores_forward) + 1), scores_forward, marker='o', linestyle='-')
plt.xlabel("Features added (in order)")
plt.ylabel("Model Accuracy")
plt.title("Forward Feature Selection Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig("forward_selection_plot.png")


selected_columns = ['Timestamp',
    'cat_glucose_value (mmol/l)',
    'temp_pattern_Alarm_No alarm(b)Insulinetype_Busy Scheduled',
    'Insulinetype_Busy Temporary',
    'glucose_diff',
    'target_majority_category'
]
df_train[selected_columns].to_csv("selected_train_svm.csv", index=False)
df_test[selected_columns].to_csv("selected_test_svm.csv", index=False)