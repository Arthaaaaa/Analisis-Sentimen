import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ======== BACA DATA =========
df = pd.read_csv("komentar_labeled_preprocessed.csv")
df.dropna(subset=["komentar_bersih", "sentimen"], inplace=True)

# ======== SIAPKAN TEKS UNTUK MASING-MASING SENTIMEN =========
positif_text = " ".join(df[df["sentimen"] == "positif"]["komentar_bersih"])
netral_text = " ".join(df[df["sentimen"] == "netral"]["komentar_bersih"])
negatif_text = " ".join(df[df["sentimen"] == "negatif"]["komentar_bersih"])

# ======== FUNGSI UNTUK MENAMPILKAN WORDCLOUD =========
def tampilkan_wordcloud(text, judul, warna="black"):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=warna,
        colormap="Set2",
        max_words=200
    ).generate(text)

    wordcloud.to_file(f"wordcloud_{judul.lower().replace(' ', '_')}.png")


    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(judul, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ======== TAMPILKAN WORDCLOUD UNTUK MASING-MASING SENTIMEN =========
tampilkan_wordcloud(positif_text, "WordCloud Komentar Positif", warna="white")
tampilkan_wordcloud(netral_text, "WordCloud Komentar Netral", warna="white")
tampilkan_wordcloud(negatif_text, "WordCloud Komentar Negatif", warna="white")
