from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv("selected_train_rf.csv")
test_df = pd.read_csv("selected_test_rf.csv")

# Define features and target
feature_cols = [
    "cat_glucose_value (mmol/l)",
    "temp_pattern_Alarm_No alarm(b)Insulinetype_Busy Scheduled",
    "Insulinetype_Busy Temporary",
    "glucose_diff"
]
target_col = "target_majority_category"

X_train = train_df[feature_cols]
y_train = train_df[target_col].astype(int)
X_test = test_df[feature_cols]
y_test = test_df[target_col].astype(int)

# Use professor’s class
learner = ClassificationAlgorithms()
evaluator = ClassificationEvaluation()

# Train & predict
pred_y_train, pred_y_test, prob_train_y, prob_test_y = learner.support_vector_machine_with_kernel(
    X_train, y_train, X_test, gridsearch=True, print_model_details=True
)

# print(sorted(y_test.unique()))
# print(sorted(pred_y_test))

# Evaluate
print("Test Accuracy:", evaluator.accuracy(y_test, pred_y_test))
print("F1 Score per class:", evaluator.f1(y_test, pred_y_test))


# Detailed classification report
report = classification_report(y_test, pred_y_test, output_dict=True)
print("Classification report:")
for label, metrics in report.items():
    print(f"{label}: {metrics}")

# Confusion matrix
cm = confusion_matrix(y_test, pred_y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()
