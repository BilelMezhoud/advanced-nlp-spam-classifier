
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from gensim.models import KeyedVectors
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import torch
import joblib
import string
import transformers
from transformers import BertTokenizer, BertModel
import fasttext

df = pd.read_csv("Data/spam.csv")
df.head(5)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    text = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    text = re.sub(r'£|\$', 'moneysymb', text)
    text = re.sub(r'\b(\+\d{1,2}\s)?\d{3}\s?\d{3}\s?\d{4}\b', 'phonenumbr', text)
    text = re.sub(r'\d+(\.\d+)?', 'numbr', text)
    text = re.sub(r'[^\w\d\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text.strip())
    return text

df['Message'] = df['Message'].apply(lambda x: preprocess_text(x))

# Create balanced dataset by oversampling the majority class
num_ham = df['Category'].value_counts()['ham']
df_spam = df[df['Category'] == 'spam'].sample(num_ham, replace=True, random_state=42)
df_ham = df[df['Category'] == 'ham']
df_balanced = pd.concat([df_ham, df_spam])
df_balanced['spam'] = df_balanced['Category'].apply(lambda x: 1 if x=='spam' else 0)

# Plot the class distribution
plt.bar(['Spam', 'Ham'], [df_balanced['spam'].sum(), len(df_balanced) - df_balanced['spam'].sum()])
plt.title('Distribution of examples in balanced dataset')
plt.xlabel('Class')
plt.ylabel('Number of examples')
plt.show()

def load_word2vec_model():
    word2vec_model = KeyedVectors.load_word2vec_format("Pre_trained_models/GoogleNews-vectors-negative300.bin", binary=True)
    return word2vec_model

def load_fasttext_model():
    ft_model = fasttext.load_model('Pre_trained_models/cc.en.300.bin')
    return ft_model

def load_glove_model():
    embeddings_index = {}
    with open("Pre_trained_models/glove.6B.300d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def vectorize_text(text: str, model_type: str) -> np.ndarray:
    if model_type == "word2vec":
        word2vec_model = load_word2vec_model()
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_tokens = [WordNetLemmatizer().lemmatize(w) for w in word_tokens if not w in stop_words]
        word_vectors = [word2vec_model[word] for word in filtered_tokens if word in word2vec_model]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)

    elif model_type == "glove":
        embeddings_index = load_glove_model()
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
        words = [WordNetLemmatizer().lemmatize(word) for word in words]
        vectors = [embeddings_index[word] for word in words if word in embeddings_index]
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    elif model_type == "fasttext":
        ft_model = load_fasttext_model()
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_tokens = [WordNetLemmatizer().lemmatize(w) for w in word_tokens if not w in stop_words]
        return ft_model.get_sentence_vector(' '.join(filtered_tokens))

    elif model_type == "bert":
        tokenizer, model = load_bert_model()
        tokens = tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            embeddings = model(input_ids)[0][0]
        return torch.mean(embeddings, dim=0).numpy()

    else:
        raise ValueError(f"Invalid model type: {model_type}")

# Load the saved SVM models
model_w2v_svm = joblib.load("w2v_svm_model.pkl")
model_Ft_svm = joblib.load("Ft_svm.pkl")
model_glove_svm = joblib.load("glove_svm_model.pkl")
model_bert_svm = joblib.load("bert_svm_model.pkl")

def predict_spam(message, model_name):
    preprocessed_text = preprocess_text(message)
    if model_name == "Word2Vec + SVM":
        model = model_w2v_svm
        vectorized_text = vectorize_text(preprocessed_text, 'word2vec')
    elif model_name == "FastText + SVM":
        model = model_Ft_svm
        vectorized_text = vectorize_text(preprocessed_text, 'fasttext')
    elif model_name == "GloVe + SVM":
        model = model_glove_svm
        vectorized_text = vectorize_text(preprocessed_text, 'glove')
    elif model_name == "BERT + SVM":
        model = model_bert_svm
        vectorized_text = vectorize_text(preprocessed_text, 'bert')
    else:
        raise ValueError("Model name not recognized")
    return model.predict(vectorized_text.reshape(1, -1))[0]
