import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Chapter7.FeatureSelection import FeatureSelectionClassification

def analyse_features(df_train, df_test, max_features, method):
    df_train = pd.read_csv("new data/all_features_train.csv",parse_dates=["Timestamp"])
    df_test = pd.read_csv("new data/all_features_test.csv",parse_dates=["Timestamp"])

    #NaN values that are now present are because of de devision by 0 during feature selection
    #So we can fill them in with 0
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    X_train = df_train.drop(columns=['target_majority_category'])
    X_train = X_train.select_dtypes(include=[np.number])
    y_train = df_train['target_majority_category']
    X_test = df_test.drop(columns=['target_majority_category'])
    X_test = X_test.select_dtypes(include=[np.number])
    y_test = df_test['target_majority_category']

    fs = FeatureSelectionClassification()

    #Forward selection
    selected_forward, ordered_forward, scores_forward = fs.forward_selection(
        max_features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        algorithm = method,
        gridsearch=False
    )

    print("Forward selected features:", selected_forward)
    print("Forward scores:", scores_forward)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores_forward) + 1), scores_forward, marker='o', linestyle='-')
    plt.xlabel("Features added (in order)")
    plt.ylabel("Model Accuracy")
    plt.title(f"Forward Feature Selection Performance {method}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"new data/forward_selection_{method}.png")

    return selected_forward, scores_forward



df_train = pd.read_csv("new data/all_features_train.csv",parse_dates=["Timestamp"])
df_test = pd.read_csv("new data/all_features_test.csv",parse_dates=["Timestamp"])

#Ran once to extract features.
#selected_forward, scores_forward = analyse_features(df_train,df_test,8,'random_forest')

selected_columns = ['Timestamp',
    'Glucose value (mmol/l)', 'glucose_diff', 
    'temp_pattern_Insulinetype_Busy Scheduled(c)Only basal', 
    'temp_pattern_glucose_trend_increasing(b)Insulinetype_Busy Scheduled', 
    'temp_pattern_Insulinetype_bolus_No Bolus(c)glucose_trend_decreasing',
    'target_majority_category'
]
df_train[selected_columns].to_csv("new data/selected_train_RF.csv", index=False)
df_test[selected_columns].to_csv("new data/selected_test_RF.csv", index=False)

#Ran once to extract features.
#analyse_features(df_train,df_test,8,'svm_rbf')
selected_columns = ['Timestamp',
    'Glucose value (mmol/l)', 'glucose_diff', 
    'min_Glucose value (mmol/l)', 'cat_glucose_value (mmol/l)_4', 
    'temp_pattern_Insulinetype_Busy Scheduled(b)Only basal',
    'target_majority_category'
]
df_train[selected_columns].to_csv("new data/selected_train_SVM.csv", index=False)
df_test[selected_columns].to_csv("new data/selected_test_SVM.csv", index=False)