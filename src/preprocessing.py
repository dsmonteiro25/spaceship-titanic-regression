import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Scaler global (apenas para referência em treino)
scaler = StandardScaler()

def preprocess_data(df, scaler=None, is_train=True):
    df = df.copy()

    # Converter booleanos para 0/1
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0})
    df['VIP'] = df['VIP'].map({True: 1, False: 0})
    df['CryoSleep'] = df['CryoSleep'].fillna(0)
    df['VIP'] = df['VIP'].fillna(0)

    # Preencher colunas numéricas
    df['Age'] = df['Age'].fillna(df['Age'].median())
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0)

    # Preencher categóricas
    df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
    df['Destination'] = df['Destination'].fillna('Unknown')

    # One-hot encoding
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'], drop_first=True)

    # Guardar PassengerId (se existir)
    passenger_ids = df['PassengerId'] if 'PassengerId' in df.columns else None

    # Remover colunas desnecessárias
    df = df.drop(columns=['Name', 'Cabin', 'PassengerId'], errors='ignore')

    # Normalizar dados numéricos
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    if is_train:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, passenger_ids, scaler
    else:
        if scaler is None:
            raise ValueError("O scaler precisa ser fornecido no modo de predição (is_train=False).")
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df, passenger_ids
