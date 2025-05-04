from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import pandas as pd

# Setup driver
path_to_driver = r"D:\chromedriver-win64\chromedriver.exe"  # Ganti path sesuai lokasi
service = Service(executable_path=path_to_driver)
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=service, options=options)

# Login manual dulu
print("Silakan login ke Twitter di browser...")
driver.get("https://twitter.com/login")
input("Setelah login selesai, tekan ENTER di terminal ini...")

# Masuk ke hasil pencarian
query = "kb pria"
search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
driver.get(search_url)
time.sleep(15)

# Inisialisasi
data = []
scroll_pause = 4
max_scrolls = 5000
last_height = driver.execute_script("return document.body.scrollHeight")

for scroll in range(max_scrolls):
    print(f"📜 Scroll ke-{scroll+1}")
    time.sleep(15)  # beri waktu ekstra agar tweet sempat muncul

    tweet_blocks = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')

    for tweet in tweet_blocks:
        try:
            WebDriverWait(tweet, 3).until(
            EC.presence_of_element_located((By.XPATH, './/div[@lang]'))
            )
            text_elem = tweet.find_element(By.XPATH, './/div[@lang]')
            text = text_elem.text

            time_elem = tweet.find_element(By.XPATH, './/time')
            timestamp = time_elem.get_attribute("datetime")

            if text.strip() != "":
               data.append({
                "platform": "twitter",
                "komentar": text,
                "timestamp": timestamp,
                "sentimen": ""
            })

        except Exception as e:
          continue

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Simpan ke CSV
df = pd.DataFrame(data)
df.to_csv("komentar_twitter_seleniumv16.csv", index=False)
print(f"✅ Berhasil simpan {len(df)} komentar ke komentar_twitter_seleniumv16.csv")

driver.quit()
