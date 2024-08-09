import pandas as pd

# Pfad zur CSV-Datei
input_csv = './data/Weather_Data_Germany.csv'

# Spalten, die extrahiert werden sollen
columns = ['longitude', 'latitude', 'msl', 'tcc', 'time', 'cdir', 'ssr', 'tsr', 'sund', 'u10', 'v10', 'u100', 'v100', 't2m']

# CSV-Datei laden
df = pd.read_csv(input_csv)

# Nur die gewünschten Spalten extrahieren
df = df[columns]

# Regionen nach 'longitude' und 'latitude' gruppieren und separate CSVs speichern
for (lon, lat), group in df.groupby(['longitude', 'latitude']):
    region_df = group.reset_index(drop=True)
    output_csv = f'./data/regions/region_{lon}_{lat}.csv'
    region_df.to_csv(output_csv, index=False)
