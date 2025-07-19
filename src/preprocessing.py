import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# O scaler será ajustado no treino e reaproveitado na predição
scaler = StandardScaler()

def preprocess_data(df, is_train=True):
    df = df.copy()

    # Mapear valores booleanos
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0})
    df['VIP'] = df['VIP'].map({True: 1, False: 0})
    df['CryoSleep'].fillna(0, inplace=True)
    df['VIP'].fillna(0, inplace=True)

    # Preencher colunas numéricas
    df['Age'].fillna(df['Age'].median(), inplace=True)
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col].fillna(0, inplace=True)

    # Preencher categóricas
    df['HomePlanet'].fillna('Unknown', inplace=True)
    df['Destination'].fillna('Unknown', inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'], drop_first=True)

    # Guardar PassengerId (para submissão no teste)
    passenger_ids = df['PassengerId'] if 'PassengerId' in df.columns else None

    # Remover colunas não utilizadas
    df.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=True, errors='ignore')

    # Normalizar colunas numéricas
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    if is_train:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, passenger_ids
