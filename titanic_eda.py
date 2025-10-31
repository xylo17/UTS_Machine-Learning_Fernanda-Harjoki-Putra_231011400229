# ==========================================
# Titanic Dataset EDA (Exploratory Data Analysis)
# Langkah 1-6
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Baca dataset
df = pd.read_csv("titanic.csv")

print("\n=== 1. INFORMASI DATASET ===")
print(df.info())
print(df.head())

# 2. Statistik deskriptif
print("\n=== 2. STATISTIK DESKRIPTIF ===")
print(df.describe(include='all'))

# 3. Cek missing values
print("\n=== 3. CEK MISSING VALUES ===")
print(df.isnull().sum())

# 4. Analisis distribusi target (Survived)
print("\n=== 4. DISTRIBUSI VARIABEL TARGET (Survived) ===")
print(df['Survived'].value_counts(normalize=True))
sns.countplot(x='Survived', data=df)
plt.title("Distribusi Penumpang Selamat vs Tidak Selamat")
plt.show()

# 5. Analisis hubungan antar variabel
print("\n=== 5. SURVIVAL RATE BERDASARKAN JENIS KELAMIN ===")
print(df.groupby('Sex')['Survived'].mean())

print("\n=== 5b. SURVIVAL RATE BERDASARKAN KELAS (Pclass) ===")
print(df.groupby('Pclass')['Survived'].mean())

# 6. Visualisasi tambahan
print("\n=== 6. VISUALISASI TAMBAHAN ===")
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate berdasarkan Jenis Kelamin")

plt.subplot(1,2,2)
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate berdasarkan Kelas Penumpang")

plt.tight_layout()
plt.show()

print("\n=== EDA SELESAI ===")
