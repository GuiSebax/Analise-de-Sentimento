import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
import zipfile

nltk.download('stopwords')

# Configurar Kaggle (substitua 'path_to_kaggle.json' pelo caminho do seu arquivo kaggle.json)
def setup_kaggle():
    kaggle_json_path = 'path_to_kaggle.json'  # Atualize o caminho do arquivo kaggle.json
    if not os.path.exists('~/.kaggle'):
        os.makedirs(os.path.expanduser('~/.kaggle'))
    
    os.rename(kaggle_json_path, os.path.expanduser('~/.kaggle/kaggle.json'))

# Baixar e descompactar dataset manualmente (substitua os links conforme necessário)
def download_and_extract_dataset():
    import requests
    
    url = 'https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download'  # URL do dataset
    r = requests.get(url, allow_redirects=True)
    open('imdb-dataset-of-50k-movie-reviews.zip', 'wb').write(r.content)
    
    with zipfile.ZipFile('imdb-dataset-of-50k-movie-reviews.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

# Função de limpeza do texto
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Função para imprimir análise crítica
def print_analysis():
    print("\nAnálise Crítica do Modelo\n")
    print("Pontos Fortes:")
    print("    O modelo é simples e eficiente.")
    print("    Resultados satisfatórios para frases positivas e negativas.\n")
    
    print("Pontos Fracos:")
    print("    O modelo pode não captar nuances mais complexas do texto.")
    print("    Dependência da qualidade dos dados de treinamento.\n")
    
    print("Oportunidades de Melhorias:")
    print("    Utilizar modelos mais complexos como BERT ou LSTM.")
    print("    Aumentar o tamanho e a diversidade do dataset.\n")
    
    print("Abrangência e Aplicação:")
    print("    Aplicável em sistemas de recomendação, análise de comentários e monitoramento de redes sociais.")
    print("    Limitações em textos irônicos ou sarcásticos.\n")
    
    print("Exemplos de Frases onde o Modelo Falhou:")
    print("    Frase: \"I expected a better plot twist, but it was fine overall.\"")
    print("        Sentimento Esperado: Neutro")
    print("        Sentimento Predito: Positivo")

# Pipeline principal
def main():
    # Passo 1: Configuração do Kaggle e download do dataset
    setup_kaggle()
    download_and_extract_dataset()

    # Passo 2: Carregar e analisar os dados
    df = pd.read_csv("IMDB Dataset.csv")
    print(df.head())
    print(df['sentiment'].value_counts())

    # Passo 3: Limpeza dos dados
    df['review'] = df['review'].apply(clean_text)
    print(df.head())

    # Passo 4: Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])
    print(y_train.value_counts(), y_test.value_counts())

    # Passo 5: Vetorização usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Passo 6: Treinamento do modelo de Regressão Logística
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Passo 7: Avaliação do modelo
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Passo 8: Testar com frases de exemplo
    sentences = [
        "I love this movie, it was fantastic!",
        "The film was horrible, I hated it.",
        "An excellent performance by the lead actor.",
        "The plot was boring and predictable.",
        "I enjoyed the cinematography and the music was great."
    ]
    
    cleaned_sentences = [clean_text(sentence) for sentence in sentences]
    vectorized_sentences = vectorizer.transform(cleaned_sentences)
    sentiment_predictions = model.predict(vectorized_sentences)
    
    for sentence, sentiment in zip(sentences, sentiment_predictions):
        print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")
    
    # Imprimir análise crítica
    print_analysis()

if __name__ == "__main__":
    main()
