import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt

# === Load and preprocess ===
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    exclude = ['Timestamp', 'target_majority_category']
    features = [col for col in train_df.columns if col not in exclude]
    return train_df, test_df, features

def create_sequences(df, features, sequence_length):
    x, y = [], []
    df = df.dropna(subset=features + ['target_majority_category'])
    for i in range(len(df) - sequence_length):
        seq = df[features].iloc[i:i + sequence_length].values
        label = df['target_majority_category'].iloc[i + sequence_length]
        x.append(seq)
        y.append(label)
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
train_path = "data/selected_train_rf.csv"
test_path = "data/selected_test_rf.csv"
train_df, test_df, features = load_data(train_path, test_path)
x_train, y_train = create_sequences(train_df, features, sequence_length)
x_test, y_test = create_sequences(test_df, features, sequence_length)
num_classes = len(np.unique(y_train))

# Compute class weights
weights = class_weight.compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_train),
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
