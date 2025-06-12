import numpy as np
import pandas as pd
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation

df_train = pd.read_csv("all_features_train.csv",parse_dates=["Timestamp"])
df_test = pd.read_csv("all_features_test.csv",parse_dates=["Timestamp"])

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

def create_target(df):
    future_window = 6

    df['target_majority_category'] = (
        df['cat_glucose_value (mmol/l)']
        .shift(-future_window + 1)  # shift upward to align current row with future window
        .rolling(window=future_window)
        .apply(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    )

    # Drop rows where label couldn't be calculated (e.g. near the end)
    df = df.dropna(subset=['target_majority_category'])

    return df

df_train = create_target(df_train)
df_test = create_target(df_test)

# Define features and target
X_train = df_train.drop(columns=['target_majority_category'])
X_train = X_train.select_dtypes(include=[np.number])
y_train = df_train['target_majority_category']
X_test = df_test.drop(columns=['target_majority_category'])
X_test = X_test.select_dtypes(include=[np.number])
y_test = df_test['target_majority_category']

# Initialize the feature selection class
fs = FeatureSelectionClassification()

# Perform forward selection (e.g., selecting 10 best features)
selected_forward, ordered_forward, scores_forward = fs.forward_selection(
    max_features=10,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    gridsearch=False
)

print("Forward selected features:", selected_forward)
print("Forward scores:", scores_forward)

# Perform backward selection (e.g., reduce to 10 best features)
print("Begin backward")
selected_backward = fs.backward_selection(
    max_features=10,
    X_train=X_train,
    y_train=y_train
)
print("End backward")
print("Backward selected features:", selected_backward)

#evalutate differente features
ca = ClassificationAlgorithms()
ce = ClassificationEvaluation()

# Evaluate forward
pred_train, pred_test, _, _ = ca.decision_tree(X_train[selected_forward], y_train, X_test[selected_forward])
print("Forward selection accuracy:", ce.accuracy(y_test, pred_test))

# Evaluate backward
pred_train, pred_test, _, _ = ca.decision_tree(X_train[selected_backward], y_train, X_test[selected_backward])
print("Backward selection accuracy:", ce.accuracy(y_test, pred_test))
