import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tcn import TCN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# === Load and preprocess ===
def load_data(train_path):
    df = pd.read_csv(train_path)
    df.columns = df.columns.str.strip()  # Important!
    features = [
        'glucose_diff',
        'min_Glucose value (mmol/l)',
        'Glucose value (mmol/l)',
        'max_Carb ratio',
        'std_Carbohydrates (g)'
    ]
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Column '{f}' not found in dataset. Check spelling and spacing.")
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
def build_tcn_model(input_shape, num_classes, nb_filters, kernel_size, dropout_rate, nb_stacks, optimizer):
    model = Sequential([
        TCN(input_shape=input_shape,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=(1, 2, 4, 8),
            dropout_rate=dropout_rate,
            nb_stacks=nb_stacks,
            use_skip_connections=True,
            use_batch_norm=True,
            return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === Main loop ===
print("hallo")
sequence_length = 6
train_path = "new data/selected_train_TCN.csv"
train_df, features = load_data(train_path)
print("loaded")
train_df = train_df.head(10000)

x_all, y_all = create_sequences(train_df, features, sequence_length)

train_df.columns = train_df.columns.str.strip()  # remove leading/trailing spaces
print("Cleaned column names:")
print(train_df.columns.tolist())


print("Number of features:", len(features))
print("Feature names:", features)

# Chronological split (no data leakage)
split_idx = int(0.8 * len(x_all))
x_train, x_val = x_all[:split_idx], x_all[split_idx:]
y_train, y_val = y_all[:split_idx], y_all[split_idx:]

num_classes = len(np.unique(y_all))

filters_list = [32, 64, 128, 256]
kernel_list = [3]
dropout_list = [0.1]
stacks_list = [1, 2]
lr_list = [0.001]
batch_sizes = [32]

results = []

i = 0

for nb_stacks in stacks_list:
    for learning_rate in lr_list:
        for batch_size in batch_sizes:
            for nb_filters in filters_list:
                for kernel_size in kernel_list:
                    for dropout_rate in dropout_list:
                        print(f"Training TCN: filters={nb_filters}, kernel={kernel_size}, dropout={dropout_rate}, stacks={nb_stacks}, lr={learning_rate}, batch={batch_size}")

                        i += 1
                        print(i)
                        
                        optimizer = Adam(learning_rate=learning_rate)

                        model = build_tcn_model(
                            input_shape=(sequence_length, x_train.shape[2]),
                            num_classes=num_classes,
                            nb_filters=nb_filters,
                            kernel_size=kernel_size,
                            dropout_rate=dropout_rate,
                            nb_stacks=nb_stacks,
                            optimizer=optimizer
                        )

                        early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
                        model.fit(x_train, y_train,
                                  epochs=50,
                                  validation_data=(x_val, y_val),

                                  batch_size=batch_size,
                                  callbacks=[early_stop],
                                  verbose=0)

                        y_val_pred = np.argmax(model.predict(x_val), axis=1)
                        f1 = f1_score(y_val, y_val_pred, average='macro')
                        print(f"→ Macro F1: {f1:.4f}")

                        results.append((nb_stacks, learning_rate, batch_size,
                                        nb_filters, kernel_size, dropout_rate, f1))
                        
# Sort results
results.sort(key=lambda x: x[6], reverse=True)  # f1-score is now the 6th element

print("\nTop 3 configs by macro F1:")
for r in results[:3]:
    print(f"stacks={r[0]}, lr={r[1]}, batch={r[2]}, filters={r[3]}, kernel={r[4]}, dropout={r[5]} → F1={r[6]:.4f}")

