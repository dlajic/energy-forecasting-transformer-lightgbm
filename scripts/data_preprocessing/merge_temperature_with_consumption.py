import os
import pandas as pd
from scripts.config_main import TEMPERATURE_RAW_PATH, TEMPERATURE_PROCESSED_PATH, ENERGY_RAW_COMED_PATH, MERGED_RAW_PATH

# === 1. Temperaturdaten einlesen und bereinigen ===
df_temp_raw = pd.read_csv(TEMPERATURE_RAW_PATH)

# TMP-Spalte bereinigen und Celsius-Werte extrahieren
df_temp_raw["TMP_CLEAN"] = (
    df_temp_raw["TMP"]
    .str.replace(",", ".", regex=False)  # deutsches Komma durch Punkt ersetzen
    .str.extract(r"([-+]?\d+\.?\d*)")[0]
    .astype(float)
)

# Unrealistische Ausreißer entfernen (9999.9)
df_temp_raw = df_temp_raw[df_temp_raw["TMP_CLEAN"] < 900].copy()

# Umrechnung in Grad Celsius
df_temp_raw["temperature_c"] = df_temp_raw["TMP_CLEAN"] / 10

# Zeitstempel bereinigen
df_temp_raw["datetime"] = pd.to_datetime(df_temp_raw["DATE"])
df_temp_raw["datetime_rounded"] = df_temp_raw["datetime"].dt.floor("h")

# Auf Stunde mitteln (z. B. mehrere Einträge pro Stunde)
df_temp_hourly = (
    df_temp_raw
    .groupby("datetime_rounded")["temperature_c"]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"datetime_rounded": "datetime"})
)

df_temp_hourly.to_csv(TEMPERATURE_PROCESSED_PATH, index=False)
print(f"Temp_preprocessed stored in: {TEMPERATURE_PROCESSED_PATH}")

# === 2. Energiedaten einlesen und aufbereiten ===
df_energy = pd.read_csv(ENERGY_RAW_COMED_PATH, sep=";")

# Umbenennen und Zeitstempel parsen
df_energy = df_energy.rename(columns={
    "Datetime": "datetime",
    "COMED_MW": "consumption_MW"
})
df_energy["datetime"] = pd.to_datetime(df_energy["datetime"], format="%d.%m.%Y %H:%M")

# sort
df_energy = df_energy.sort_values("datetime").reset_index(drop=True)

# === 3. Merge ===
df_merged = pd.merge(
    df_energy,
    df_temp_hourly,
    on="datetime",
    how="left"
)

# === 4. Speichern ===
df_merged.to_csv(MERGED_RAW_PATH, index=False)
print(f"Merged Data gespeichert unter: {MERGED_RAW_PATH}")
