# src/predict.py

import pandas as pd
import joblib
from preprocessing import preprocess_data

# Carregar dados de teste
df_test = pd.read_csv('data/raw/test.csv')

# Carregar modelo treinado
model = joblib.load('models/logistic_regression.pkl')

# Pré-processar os dados de teste
X_test_processed, passenger_ids = preprocess_data(df_test, is_train=False)

# Gerar previsões
y_pred = model.predict(X_test_processed)
y_pred_bool = y_pred.astype(bool)

# Montar submissão
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': y_pred_bool
})

submission.to_csv('submissions/submission.csv', index=False)
print("Submissão salva em: submissions/submission.csv")
