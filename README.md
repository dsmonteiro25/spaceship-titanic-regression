# Spaceship Titanic - Regressão Logística

Projeto acadêmico da disciplina de Inteligligência Artificial que implementa uma solução para o desafio Spaceship Titanic do Kaggle, utilizando regressão logística.

---

## 📁 Estrutura do Projeto

```
spaceship-titanic-regression/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/                # (opcional)
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/                       # Modelo treinado
│   └── logistic_regression.pkl
│
├── src/                          # Scripts principais
│   ├── preprocessing.py          # Função de pré-processamento
│   ├── train.py                  # Treinamento do modelo
│   └── predict.py                # Geração de submissão
│
├── submissions/
│   └── submission.csv            # Arquivo pronto para o Kaggle
│
├── README.md
└── requirements.txt
```

---

## 🚀 Como usar

### 1. Baixe os dados

Baixe `train.csv`, `test.csv` e `sample_submission.csv` no [Kaggle - Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data)  
e coloque-os na pasta `data/raw/`.

---

### 2. Configure o ambiente

Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate       # Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

### 3. Treine o modelo

```bash
python src/train.py
```

Este script:
- Pré-processa os dados de treino
- Treina um modelo de regressão logística
- Salva o modelo em `models/logistic_regression.pkl`

---

### 4. Gere a submissão

```bash
python src/predict.py
```

Este script:
- Pré-processa os dados de teste
- Carrega o modelo treinado
- Gera o arquivo `submissions/submission.csv` para enviar ao Kaggle

---

### 5. Submeter ao Kaggle

Faça upload do arquivo `submissions/submission.csv` na plataforma da competição.

---

## 📧 Contato

Desenvolvido por **Daniel Monteiro, Edson Carlos, Wendson Menezes e Geovanna**  
Email: danielsm05862@gmail.com

---

## 🪪 Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo LICENSE para mais detalhes.
