import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# ======== BACA FILE CSV KAMU =========
df = pd.read_csv("dataset_IG.csv")  # GANTI kalau nama file beda
df = df.dropna(subset=["komentar"])  # Buang komentar kosong
texts = df["komentar"].tolist()

# ======== LOAD MODEL IndoBERT SENTIMEN =========
model_name = "mdhugol/indonesia-bert-sentiment-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Pipeline untuk klasifikasi
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ======== KLASIFIKASI KOMENTAR =========
results = classifier(texts, truncation=True, max_length=512, batch_size=8)

# Mapping label model ke label teks
label_map = {
    "LABEL_0": "positif",
    "LABEL_1": "netral",
    "LABEL_2": "negatif"
}
df["sentimen"] = [label_map[r["label"]] for r in results]

# ======== SIMPAN HASIL =========
df.to_csv("dataset_IG_labeled.csv", index=False)
print("✅ Berhasil! Komentar dengan label sentimen disimpan di 'komentar_X_labeled_terbaru.csv'")
