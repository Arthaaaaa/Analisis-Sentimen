import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download("stopwords")

# Load data
df = pd.read_csv("komentar_labeled.csv")
df = df.dropna(subset=["komentar", "sentimen"])

# Inisialisasi stemmer & stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words("indonesian"))

def clean_text(text):
    # Case folding
    text = text.lower()
    # Hapus URL
    text = re.sub(r"http\S+|www.\S+", "", text)
    # Hapus angka dan tanda baca
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenisasi manual (lebih aman)
    tokens = text.split()
    # Stopword removal
    filtered = [w for w in tokens if w not in stop_words]
    # Stemming
    stemmed = [stemmer.stem(word) for word in filtered]
    return " ".join(stemmed)


# Terapkan ke semua komentar
df["komentar_bersih"] = df["komentar"].apply(clean_text)

# Simpan hasil
df.to_csv("komentar_labeled_preprocessed.csv", index=False)
print("✅ Preprocessing selesai. Simpan ke komentar_labeled_preprocessed.csv")
