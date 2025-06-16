# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.class_weight import compute_sample_weight
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your pre-split train and test data
# train_df = pd.read_csv("selected_train_rf.csv")
# test_df = pd.read_csv("selected_test_rf.csv")


# X_train = train_df.drop(columns=["target_majority_category"])
# y_train = train_df["target_majority_category"]
# X_test = test_df.drop(columns=["target_majority_category"])
# y_test = test_df["target_majority_category"]

# X_train = X_train.drop(columns=["Timestamp"], errors="ignore")
# X_test = X_test.drop(columns=["Timestamp"], errors="ignore")


# # Calculate class weights for imbalance
# sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# param_grid = {
#     'n_estimators': [100, 200, 300],              # Try more trees
#     'max_depth': [5, 10, 15, None],               # Add deeper trees and unlimited depth
#     'min_samples_split': [2, 5, 10],              # Allow splits requiring more samples
#     'min_samples_leaf': [1, 2, 5],                # Allow larger minimum leaf sizes
# }


# rf = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train) 


# # Best model
# best_rf = grid_search.best_estimator_

# # Predict
# y_pred = best_rf.predict(X_test)

# # Evaluation
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred, output_dict=True)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - Random Forest")
# plt.tight_layout()
# plt.show()

# # Print best parameters and classification report
# print("Best parameters found:", grid_search.best_params_)
# print("\nClassification report:")
# for label, metrics in class_report.items():
#     print(f"{label}: {metrics}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline

# Load your pre-split train and test data
train_df = pd.read_csv("selected_train_rf.csv")
test_df = pd.read_csv("selected_test_rf.csv")

X_train = train_df.drop(columns=["target_majority_category", "Timestamp"], errors="ignore")
y_train = train_df["target_majority_category"]
X_test = test_df.drop(columns=["target_majority_category", "Timestamp"], errors="ignore")
y_test = test_df["target_majority_category"]

# Define class weights manually
class_weights = {0: 10, 1: 5, 2: 1, 3: 1, 4: 1}

# Define pipeline (no oversampling)
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42, class_weight=class_weights))
])


# Define parameter grid (prefix with 'rf__' for grid search to access RandomForest params)
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [5, 10, 15, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 5],
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest with Oversampling")
plt.tight_layout()
plt.show()

# Print best parameters and classification report
print("Best parameters found:", grid_search.best_params_)
print("\nClassification report:")
for label, metrics in class_report.items():
    print(f"{label}: {metrics}")
