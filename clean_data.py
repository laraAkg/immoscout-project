import pandas as pd
import re

# JSON laden
df = pd.read_json('immo_spider/immoscout_listings.json')

# Spalten bereinigen
def clean_rooms(room_str):
    if pd.notnull(room_str):
        match = re.search(r'[\d.]+', room_str)
        return float(match.group()) if match else None
    return None

def clean_size(size_str):
    if pd.notnull(size_str):
        match = re.search(r'[\d.]+', size_str)
        return float(match.group()) if match else None
    return None

def clean_price(price_str):
    if pd.notnull(price_str):
        price_str = price_str.replace("’", "").replace("CHF", "").replace(".–", "")
        match = re.search(r'\d+', price_str)
        return int(match.group()) if match else None
    return None

# Bereinigung anwenden
df['rooms'] = df['rooms'].apply(clean_rooms)
df['size_m2'] = df['size'].apply(clean_size)
df['price_chf'] = df['price'].apply(clean_price)

# "location" in Straße + PLZ trennen (optional)
df[['street', 'plz']] = df['location'].str.extract(r'^(.*),\s*(\d{4}\s+\w+)')

# Originalspalten löschen (optional)
df = df.drop(columns=['size', 'price'])

print(df.head())

# Optional speichern
df.to_csv('immoscout_clean.csv', index=False)
