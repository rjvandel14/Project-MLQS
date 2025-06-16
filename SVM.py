from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation

import pandas as pd
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv("data/selected_train_SVM.csv")
test_df = pd.read_csv("data/selected_test_SVM.csv")

# Define features and target
feature_cols = [
    "Glucose value (mmol/l)",
    "mean_Glucose value (mmol/l)",
    "median_Insuline units (basal)",
    "temp_pattern_Insulinetype_Scheduled(b)Only basal"
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

print(sorted(y_test.unique()))
#print(sorted(pred_y_test.unique()))

# Evaluate
print("Test Accuracy:", evaluator.accuracy(y_test, pred_y_test))
print("F1 Score per class:", evaluator.f1(y_test, pred_y_test))
print("Precision per class:", evaluator.precision(y_test, pred_y_test))
print("Recall per class:", evaluator.recall(y_test, pred_y_test))


# Confusion matrix
cm = evaluator.confusion_matrix(y_test, pred_y_test, labels=[0, 1, 2, 3, 4])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()
