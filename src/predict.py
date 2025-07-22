import pandas as pd
import joblib
import os
from preprocessing import preprocess_data

# Carregar dados de teste
df_test = pd.read_csv('data/raw/test.csv')

# Carregar modelo e scaler treinados
model = joblib.load('models/logistic_regression_best.pkl')
scaler = joblib.load('models/scaler.pkl')

# Pré-processar os dados de teste com o scaler carregado
X_test_processed, passenger_ids = preprocess_data(df_test, scaler=scaler, is_train=False)

# Gerar previsões
y_pred = model.predict(X_test_processed)
y_pred_bool = y_pred.astype(bool)

# Gerar submissão
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': y_pred_bool
})

#salvar arquivo

submission.to_csv('data/raw/submission.csv', index=False)
print("Submissão salva em: raw/submission.csv")
