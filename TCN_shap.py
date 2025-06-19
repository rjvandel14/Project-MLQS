import shap
import numpy as np
import pandas as pd
from tcn import TCN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# === Load and preprocess ===
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    train_df = train_df.select_dtypes(include=[np.number])
    test_df = pd.read_csv(test_path)
    test_df = test_df.select_dtypes(include=[np.number])
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    exclude = ['Timestamp', 'target_majority_category']
    features = [col for col in train_df.columns if col not in exclude]
    return train_df, test_df, features

def create_sequences(df, features, sequence_length):
    x, y = [], []
    df = df.dropna(subset=['target_majority_category'])
    feature_array = df[features].values
    target_array = df['target_majority_category'].values

    for i in range(len(df) - sequence_length):
        x.append(feature_array[i:i + sequence_length])
        y.append(target_array[i + sequence_length])

    return np.array(x), np.array(y)

# === Build final TCN model ===
def build_tcn_model(input_shape, num_classes):
    model = Sequential([
        TCN(input_shape=input_shape,
            nb_filters=64,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            dropout_rate=0.2,
            nb_stacks=2,
            use_skip_connections=True,
            use_batch_norm=True,
            return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === Run final model ===
sequence_length = 6
train_path = "new data/all_features_train.csv"
test_path = "new data/all_features_test.csv"
# train_path = "new data/selected_train_SVM.csv"
# test_path = "new data/selected_test_SVM.csv"
train_df, test_df, features = load_data(train_path, test_path)
x_train, y_train = create_sequences(train_df, features, sequence_length)
x_test, y_test = create_sequences(test_df, features, sequence_length)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
num_classes = len(np.unique(y_train))
print(np.unique(y_train), y_train.dtype)
# Compute class weights
weights = class_weight.compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_train).astype(int),
                                             y=y_train)
class_weights = dict(enumerate(weights))

model = build_tcn_model((sequence_length, x_train.shape[2]), num_classes)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=50,
          batch_size=32,
          callbacks=[early_stop],
          class_weight=class_weights,
          verbose=1)

y_pred = np.argmax(model.predict(x_test), axis=1)
print("Classification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Final TCN Model")
plt.show()

# === Select background and test samples ===
# Limit size for performance (especially with DeepExplainer)
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
explainer = shap.GradientExplainer(model, background)

# Select a test subset to explain
x_explain = x_test[:100]  # instead of the whole x_test
shap_values = explainer.shap_values(x_explain)

# Average over time steps: (samples, time_steps, features) → (samples, features)
shap_values_avg = [sv.mean(axis=1) for sv in shap_values]
x_explain_avg = x_explain.mean(axis=1)

# === Calculate mean absolute SHAP value per feature ===
mean_abs_shap = np.mean(np.abs(shap_values_avg[0]), axis=0)  # (features,)
feature_importance = pd.Series(mean_abs_shap, index=features).sort_values(ascending=False)

# === Show top features ===
top_n = 10  # choose number of features
print("\nTop {} features by SHAP importance:".format(top_n))
print(feature_importance.head(top_n))

top_features = feature_importance.head(top_n).index.tolist()
print("top features: ", top_features)