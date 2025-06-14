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

# def create_target(df):
#     future_window = 6

#     df['target_majority_category'] = (
#         df['cat_glucose_value (mmol/l)']
#         .shift(-future_window + 1)  # shift upward to align current row with future window
#         .rolling(window=future_window)
#         .apply(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
#     )

#     # Drop rows where label couldn't be calculated (e.g. near the end)
#     df = df.dropna(subset=['target_majority_category'])

#     return df

# df_train = create_target(df_train)
# df_test = create_target(df_test)

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
    max_features=5,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
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
    'Glucose value (mmol/l)',
    'mean_Glucose value (mmol/l)',
    'mean_Insuline units (basal)',
    'temp_pattern_glucose_trend_decreasing(b)Only basal',
    'median_Insuline units (basal)',
    'target_majority_category'
]
df_train[selected_columns].to_csv("selected_train.csv", index=False)
df_test[selected_columns].to_csv("selected_test.csv", index=False)