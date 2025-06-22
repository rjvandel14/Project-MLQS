from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation

import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("new data/selected_train_SVM-1.csv")
test_df = pd.read_csv("new data/selected_test_SVM-1.csv")

feature_cols = [
    "Glucose value (mmol/l)",
    "glucose_diff",
    "min_Glucose value (mmol/l)",
    "cat_glucose_value (mmol/l)_4",
    "temp_pattern_Insulinetype_Busy Scheduled(b)Only basal"
]
target_col = "target_majority_category"

X_train = train_df[feature_cols]
y_train = train_df[target_col].astype(int)
X_test = test_df[feature_cols]
y_test = test_df[target_col].astype(int)

learner = ClassificationAlgorithms()
evaluator = ClassificationEvaluation()

class_weights = {0: 10, 1: 6, 2: 1, 3: 1, 4: 1}

pred_y_train, pred_y_test, prob_train_y, prob_test_y = learner.support_vector_machine_with_kernel(
    X_train, y_train, X_test, gridsearch=True, print_model_details=True, class_weight=class_weights
)

print("Test Accuracy:", evaluator.accuracy(y_test, pred_y_test))
print("F1 Score per class:", evaluator.f1(y_test, pred_y_test))
print("Precision per class:", evaluator.precision(y_test, pred_y_test))
print("Recall per class:", evaluator.recall(y_test, pred_y_test))

report = classification_report(y_test, pred_y_test, output_dict=True)
print("Classification report:")
for label, metrics in report.items():
    print(f"{label}: {metrics}")

cm = evaluator.confusion_matrix(y_test, pred_y_test, labels=[0, 1, 2, 3, 4])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()

print("Class distribution in training set:")
print(y_train.value_counts(normalize=True).sort_index())