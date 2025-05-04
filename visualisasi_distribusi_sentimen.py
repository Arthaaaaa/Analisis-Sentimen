import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======== BACA DATA =========
df = pd.read_csv("komentar_labeled_preprocessed.csv")
df.dropna(subset=["timestamp", "sentimen", "platform"], inplace=True)

# Konversi kolom timestamp ke datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', errors='coerce')
df.dropna(subset=["timestamp"], inplace=True)

# Tambahkan kolom tanggal dan jam
df["tanggal"] = df["timestamp"].dt.date
df["jam"] = df["timestamp"].dt.hour

# Setup tema seaborn
sns.set(style="whitegrid")

# ======== VISUALISASI DISTRIBUSI SENTIMEN =========
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="sentimen", palette="pastel", order=["negatif", "netral", "positif"])
plt.title("Distribusi Sentimen")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Komentar")
plt.tight_layout()
plt.savefig("distribusi_sentimen.png")
plt.show()

# ======== KOMENTAR PER HARI =========
komentar_per_hari = df.groupby("tanggal").size()

plt.figure(figsize=(10,5))
komentar_per_hari.plot(kind="line", marker="o", color="teal")
plt.title("Aktivitas Komentar per Hari")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Komentar")
plt.grid(True)
plt.tight_layout()
plt.savefig("komentar_per_hari.png")
plt.show()

# ======== KOMENTAR PER JAM =========
plt.figure(figsize=(8,4))
sns.countplot(data=df, x="jam", palette="coolwarm")
plt.title("Aktivitas Komentar per Jam")
plt.xlabel("Jam (0–23)")
plt.ylabel("Jumlah Komentar")
plt.tight_layout()
plt.savefig("komentar_per_jam.png")
plt.show()

# ======== KOMENTAR PER PLATFORM =========
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="platform", palette="Set2")
plt.title("Distribusi Komentar per Platform")
plt.xlabel("Platform")
plt.ylabel("Jumlah Komentar")
plt.tight_layout()
plt.savefig("komentar_per_platform.png")
plt.show()

# ======== KOMBINASI SENTIMEN vs PLATFORM =========
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="platform", hue="sentimen", palette="muted", order=df["platform"].value_counts().index)
plt.title("Distribusi Sentimen per Platform")
plt.xlabel("Platform")
plt.ylabel("Jumlah Komentar")
plt.legend(title="Sentimen")
plt.tight_layout()
plt.savefig("sentimen_per_platform.png")
plt.show()
