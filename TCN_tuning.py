import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# === Load and preprocess ===
def load_data(train_path):
    df = pd.read_csv(train_path)
    exclude = ['Timestamp', 'target_majority_category']
    features = [col for col in df.columns if col not in exclude]
    return df, features

def create_sequences(df, features, sequence_length):
    x, y = [], []
    df = df.dropna(subset=features + ['target_majority_category'])
    for i in range(len(df) - sequence_length):
        seq = df[features].iloc[i:i + sequence_length].values
        label = df['target_majority_category'].iloc[i + sequence_length]
        x.append(seq)
        y.append(label)
    return np.array(x), np.array(y)

# === Model builder ===
def build_tcn_model(input_shape, num_classes, nb_filters, kernel_size, dropout_rate):
    model = Sequential([
        TCN(input_shape=input_shape,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=(1, 2, 4, 8),
            dropout_rate=dropout_rate,
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

# === Main loop ===
sequence_length = 6
train_path = "data/selected_train_rf.csv"
train_df, features = load_data(train_path)
x_all, y_all = create_sequences(train_df, features, sequence_length)

# Split into train and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
num_classes = len(np.unique(y_all))

filters_list = [64, 128]
kernel_list = [2, 3]
dropout_list = [0.0, 0.2]

results = []

for nb_filters in filters_list:
    for kernel_size in kernel_list:
        for dropout_rate in dropout_list:
            print(f"Training TCN: filters={nb_filters}, kernel={kernel_size}, dropout={dropout_rate}")
            model = build_tcn_model((sequence_length, x_train.shape[2]), num_classes,
                                    nb_filters, kernel_size, dropout_rate)

            early_stop = EarlyStopping(patience=5, restore_best_weights=True)
            model.fit(x_train, y_train,
                      epochs=50,
                      batch_size=32,
                      callbacks=[early_stop],
                      verbose=0)

            y_val_pred = np.argmax(model.predict(x_val), axis=1)
            f1 = f1_score(y_val, y_val_pred, average='macro')

            print(f"→ Macro F1: {f1:.4f}")
            results.append((nb_filters, kernel_size, dropout_rate, f1))

# Sort results
results.sort(key=lambda x: x[3], reverse=True)

print("\nTop 3 configs by macro F1:")
for r in results[:3]:
    print(f"filters={r[0]}, kernel={r[1]}, dropout={r[2]} → F1={r[3]:.4f}")
