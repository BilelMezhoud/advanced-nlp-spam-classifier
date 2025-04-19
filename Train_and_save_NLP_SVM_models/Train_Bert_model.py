
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

# Vectorize the text data using the BERT embeddings
df_balanced['Message'] = df_balanced['Message'].apply(lambda x: vectorize_text_bert(x))

# Split the data into train and test sets and evaluate the performance of the SVM classifier
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Message'].tolist(), df_balanced['spam'], stratify=df_balanced['spam'])
bert_svm = SVC()
# perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000]}
grid_search = GridSearchCV(bert_svm, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Save the trained model
joblib.dump(grid_search.best_estimator_, 'bert_svm_model.pkl')

# Check for missing values
if np.isnan(X_test).sum().sum() > 0:
    X_test = np.nan_to_num(X_test)

# Check for infinite or very large values
if not np.isfinite(X_test).all():
    X_test = np.nan_to_num(X_test)

# Evaluate the model on the test data
predictions = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print("Accuracy score using BERT:", accuracy)
print("Precision score using BERT:", precision)
print("Recall score using BERT:", recall)
print("F1 score using BERT:", f1)