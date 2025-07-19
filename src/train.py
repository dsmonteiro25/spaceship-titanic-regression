import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocessing import preprocess_data

# Carregar dados
df = pd.read_csv('data/raw/train.csv')

# Separar X e y
y = df['Transported'].astype(int)
X = df.drop(columns=['Transported'])

# Pré-processar
X_processed, _ = preprocess_data(X, is_train=True)

# Treinar modelo
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Salvar modelo
joblib.dump(model, 'models/logistic_regression.pkl')
print("Modelo salvo em: models/logistic_regression.pkl")