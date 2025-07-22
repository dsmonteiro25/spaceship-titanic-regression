import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from preprocessing import preprocess_data
from preprocessing import scaler

#Preparar os dados
df = pd.read_csv('data/raw/train.csv')
y = df['Transported'].astype(int)
X = df.drop(columns=['Transported'])

X_processed, _, scaler = preprocess_data(X, is_train=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

#Definir hiperparâmetros
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],  # compatível com l1 e l2
    'class_weight': [None, 'balanced'],
    'max_iter': [500, 1000]
}

#Rodar Grid Search
grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

#Etapa 4: Resultados da busca
print("Melhor combinação encontrada:")
print(grid_search.best_params_)
print("\nMelhor score (F1):", grid_search.best_score_)

# Salvar resultados
os.makedirs('data/raw', exist_ok=True)
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv('data/raw/grid_search_results.csv', index=False)
print("Resultados completos salvos em: data/raw/grid_search_results.csv")

#Treinar modelo final com dados de treino
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

#Avaliação no conjunto de validação
y_pred_val = best_model.predict(X_val)
print("\nAvaliação do modelo final:")
print(classification_report(y_val, y_pred_val))

#Salvar o modelo
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/logistic_regression_best.pkl')
joblib.dump(scaler, 'models/scaler.pkl')  # se tiver salvado o scaler também
print("Modelo final salvo em: models/logistic_regression_best.pkl")
