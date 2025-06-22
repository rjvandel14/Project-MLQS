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
from sklearn.metrics import f1_score

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

def build_tcn_model(input_shape, num_classes):
    model = Sequential([
        TCN(input_shape=input_shape,
            nb_filters=256,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            dropout_rate=0.1,
            nb_stacks=1,
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

sequence_length = 6
train_path = "new data/selected_train_TCN.csv"
test_path = "new data/selected_test_TCN.csv"
train_df, test_df, features = load_data(train_path, test_path)
x_train, y_train = create_sequences(train_df, features, sequence_length)
x_test, y_test = create_sequences(test_df, features, sequence_length)
num_classes = len(np.unique(y_train))

split_idx = int(0.8 * len(x_train))
x_train_new, x_val = x_train[:split_idx], x_train[split_idx:]
y_train_new, y_val = y_train[:split_idx], y_train[split_idx:]

weights = class_weight.compute_class_weight(class_weight='balanced',
                                             classes=np.unique(y_train),
                                             y=y_train)
class_weights = dict(enumerate(weights))

class_weights2 = [
    {0: 5, 1: 5, 2: 1, 3: 1, 4: 1}
]

best_f1 = -1
best_weights = None
best_model = None
results = []

for weights in class_weights2:
    print(f"Trying class weights: {weights}")
    model = build_tcn_model((sequence_length, x_train.shape[2]), num_classes)
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(x_train_new, y_train_new,
              validation_data=(x_val, y_val),
              epochs=50,
              batch_size=32,
              callbacks=[early_stop],
              class_weight=weights,
              verbose=0)

    y_val_pred = np.argmax(model.predict(x_val), axis=1)
    f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"→ Macro F1 (val): {f1:.4f}")
    results.append((weights, f1))

    if f1 > best_f1:
        best_f1 = f1
        best_weights = weights
        best_model = model

print("\nBest weights based on validation F1:", best_weights)

# Evaluate best model on test set
y_test_pred = np.argmax(best_model.predict(x_test), axis=1)
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_test_pred, digits=3))

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Best TCN Model (Test Set)")
plt.show()