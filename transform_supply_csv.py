import pandas as pd

# CSV-Datei laden, fehlerhafte Zeilen ignorieren
df = pd.read_csv('./data/Realised_Supply_Germany.csv', sep=';')

# Date from und Date to in datetime umwandeln
df['Date from'] = pd.to_datetime(df['Date from'], format='%d.%m.%y %H:%M')
df['Date to'] = pd.to_datetime(df['Date to'], format='%d.%m.%y %H:%M')

# Ungültige Zeilen entfernen
df = df.dropna(subset=['Date from', 'Date to'])

# Eine neue Spalte für die Stunde hinzufügen
df['time'] = df['Date from'].dt.floor('h')

# Sicherstellen, dass alle relevanten Spalten numerisch sind
for column in df.columns:
    if column not in ['Date from', 'Date to', 'time']:
        df[column] = df[column].str.replace('.', '').str.replace(',', '.')
        df[column] = pd.to_numeric(df[column])

# Stündliche Werte aggregieren
hourly_df = df.groupby('time').sum().reset_index()

# Ergebnis in eine neue CSV-Datei speichern
hourly_df.to_csv('./data/Realised_Supply_Germany_Hourly.csv', index=False)

print("Die Daten wurden erfolgreich auf stündliche Werte verkürzt und gespeichert.")
