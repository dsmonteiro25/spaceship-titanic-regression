import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

# Carregar dados
df = pd.read_csv('data/raw/train.csv')
y = df['Transported'].astype(int)
X = df.drop(columns=['Transported'])

# Carregar scaler salvo
scaler = joblib.load('models/scaler.pkl')

# Pré-processar os dados (modo validação, com scaler carregado)
X_processed, _, = preprocess_data(X, scaler=scaler, is_train=False)

# Separar conjunto de validação
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Carregar modelo treinado
model = joblib.load('models/logistic_regression_best.pkl')

# Prever no conjunto de validação
y_pred = model.predict(X_val)

# Relatório de métricas
print("\nAvaliação local do modelo salvo:")
print(classification_report(y_val, y_pred))

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Modelo Salvo')
plt.tight_layout()
plt.show()
