# ==========================================
# Titanic Dataset Preprocessing
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Baca dataset
df = pd.read_csv("titanic.csv")

print("\n=== SEBELUM PREPROCESSING ===")
print(df.info())

# 2. Hapus kolom tidak relevan
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 3. Tangani missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Encoding variabel kategorikal
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\n=== SETELAH ENCODING & CLEANING ===")
print(df.head())

# 5. Pisahkan fitur dan target
X = df.drop('Survived', axis=1)
y = df['Survived']

# 6. Split data menjadi train dan test (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Normalisasi fitur (opsional tapi disarankan)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== DATA SIAP DIGUNAKAN UNTUK MODEL ===")
print("Jumlah data latih:", X_train.shape)
print("Jumlah data uji:", X_test.shape)

# 8. Simpan hasil preprocessing ke file baru
pd.DataFrame(X_train, columns=X.columns).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nFile hasil preprocessing telah disimpan:")
print(" - X_train.csv")
print(" - X_test.csv")
print(" - y_train.csv")
print(" - y_test.csv")
