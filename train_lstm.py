import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# ======== BACA DATA =========
df = pd.read_csv("komentar_labeled_preprocessed.csv")
df = df.dropna(subset=["komentar_bersih", "sentimen"])

# ======== ENCODE LABEL SENTIMEN =========
label_map = {"negatif": 0, "netral": 1, "positif": 2}
df["label"] = df["sentimen"].map(label_map)

# ======== OVERSAMPLING DATA NEGATIF =========
df_negatif = df[df["label"] == 0]
df_netral = df[df["label"] == 1]
df_positif = df[df["label"] == 2]

df_negatif_oversampled = resample(
    df_negatif,
    replace=True,
    n_samples=len(df_positif),  # samakan dengan jumlah terbesar
    random_state=42
)

df_balanced = pd.concat([df_negatif_oversampled, df_netral, df_positif])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ======== TOKENISASI =========
max_words = 10000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df_balanced["komentar_bersih"])



# Simpan tokenizer sebagai JSON
with open("tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())


sequences = tokenizer.texts_to_sequences(df_balanced["komentar_bersih"])
padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

X = padded
y = to_categorical(df_balanced["label"], num_classes=3)

# ======== SPLIT TRAIN & TEST =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======== MODEL LSTM =========
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# ======== TRAIN MODEL TANPA class_weight =========
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# ======== EVALUASI =========
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(
    y_true_labels, y_pred_labels,
    target_names=["negatif", "netral", "positif"]
))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true_labels, y_pred_labels))

# ======== SIMPAN MODEL =========
model.save("model_sentimen_lstm.h5")

# ======== SIMPAN TOKENIZER =========
import joblib
joblib.dump(tokenizer, "tokenizer.joblib")
print("✅ Tokenizer berhasil disimpan ke tokenizer.joblib")

