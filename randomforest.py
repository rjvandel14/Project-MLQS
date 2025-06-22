import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline

# Load data
train_df = pd.read_csv("new data/selected_train_RF.csv")
test_df = pd.read_csv("new data/selected_test_RF.csv")

X_train = train_df.drop(columns=["target_majority_category", "Timestamp"], errors="ignore")
y_train = train_df["target_majority_category"]
X_test = test_df.drop(columns=["target_majority_category", "Timestamp"], errors="ignore")
y_test = test_df["target_majority_category"]

# Define class weights manually
class_weights = {0: 50, 1: 20, 2: 1, 3: 1, 4: 1}


# Define pipeline with new class weights
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42, class_weight=class_weights))
])


# Define parameter grid 
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [5, 10, 15, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 5],
}

# grid search with time series split
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
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

# Best parameters and classification report
print("Best parameters found:", grid_search.best_params_)
print("\nClassification report:")
for label, metrics in class_report.items():
    print(f"{label}: {metrics}")


# class distribution
print("Class distribution in training set:")
print(y_train.value_counts(normalize=True).sort_index())

# feature importances
importances = best_model.named_steps['rf'].feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))

