import pandas as pd, numpy as np

df = pd.read_csv('C:/Users/JC/Desktop/Tesis/datasets/dataset_entrenamiento_completo_2023_2026_clean.csv')
print('=== SAMPLES PER REGION ===')
print(df['origen'].value_counts())
print()

print('=== HAB RATE PER REGION ===')
for r in df['origen'].unique():
    mask = df['origen'] == r
    hab_rate = df.loc[mask, 'es_floracion'].mean()*100
    chl_mean = df.loc[mask, 'clorofila_ugl'].mean()
    print(f"{r:20s}: n={mask.sum():>5d}, HAB={hab_rate:.1f}%, chl_mean={chl_mean:.2f}")
print()

print('=== UNIQUE DATES PER REGION ===')
for r in df['origen'].unique():
    mask = df['origen'] == r
    nd = df.loc[mask, 'fecha'].nunique()
    print(f"{r:20s}: {nd} unique dates")
print()

print('=== FEATURE STATS ===')
print(df[['B2','B3','B4','B5','B8','clorofila_ugl','es_floracion']].describe())

print()
print('=== DATA RANGE PER REGION ===')
for r in df['origen'].unique():
    mask = df['origen'] == r
    sub = df.loc[mask]
    print(f"{r:20s}: fecha [{sub['fecha'].min()}, {sub['fecha'].max()}]")
    if sub['lat'].notna().any():
        print(f"         lat [{sub['lat'].min():.4f},{sub['lat'].max():.4f}], lon [{sub['lon'].min():.4f},{sub['lon'].max():.4f}]")
