import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Baca data
df = pd.read_csv("dataset_komentar_labeled_balance_preprocessed.csv")
df = df.dropna(subset=["sentimen"])

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(6, 5))

# Hitung jumlah komentar per sentimen
sentimen_counts = df["sentimen"].value_counts().reindex(["negatif", "netral", "positif"])

# Plot bar chart
colors = ["#f87171", "#facc15", "#4ade80"]
ax = sns.barplot(x=sentimen_counts.index, y=sentimen_counts.values, palette=colors)

# Tambahkan label
for i, val in enumerate(sentimen_counts.values):
    ax.text(i, val + 5, str(val), ha='center', va='bottom', fontsize=12)

plt.title("Distribusi Sentimen Komentar", fontsize=14)
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Komentar")

# Simpan ke folder static
os.makedirs("static", exist_ok=True)
plt.tight_layout()
plt.savefig("static/visualisasi_sentimen.png")
plt.close()

print("✅ visualisasi_sentimen.png berhasil dibuat di folder static/")
