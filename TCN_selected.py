import pandas as pd

df_train = pd.read_csv("new data/all_features_train.csv",parse_dates=["Timestamp"])
df_test = pd.read_csv("new data/all_features_test.csv",parse_dates=["Timestamp"])

selected_columns = ['Timestamp',
                    'glucose_diff', 'min_Glucose value (mmol/l)', 'Glucose value (mmol/l)', 
                    'max_Carb ratio', 'std_Carbohydrates (g)',
                    'target_majority_category'
]
df_train[selected_columns].to_csv("new data/selected_train_TCN.csv", index=False)
df_test[selected_columns].to_csv("new data/selected_test_TCN.csv", index=False)