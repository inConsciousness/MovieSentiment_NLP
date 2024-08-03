# Step 1: Set up Environment
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import keras as ks
#from keras import Sequential,layers.Embedding,layers.LSTM, layers.Dense, layers.Dropout
#from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Data Collection
data = pd.read_csv('C:/Users/Asus/OneDrive/Desktop/Ideas_and_Steps/NLP_Projects/Sentiment_Analysis_project/IMDB_Dataset.csv')

# Step 3: Data Preprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = nltk.re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = nltk.re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['cleaned_review'] = data['review'].apply(preprocess)

# Step 4: Data Manipulation
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Alternatively, LSTM Model
max_words = 5000
max_len = 500
seq=ks.Sequential()
lstm_model = seq
lstm_model.add(ks.layers.Embedding(max_words, 128, input_length=max_len))
lstm_model.add(ks.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(ks.layers.Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# For LSTM, you need padded sequences
padsq=ks.preprocessing.sequence()
X_train_seq = padsq(X_train, maxlen=max_len)
X_test_seq = padsq(X_test, maxlen=max_len)

lstm_model.fit(X_train_seq, y_train, validation_split=0.2, epochs=5, batch_size=64)

# Step 6: Model Testing and Validation
# Evaluate Logistic Regression Model
y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Model")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Evaluate LSTM Model
loss, accuracy = lstm_model.evaluate(X_test_seq, y_test)
print("LSTM Model")
print("Accuracy:", accuracy)

# Step 7: Prediction
def predict_sentiment(review, model, vectorizer=None, lstm=False):
    review = preprocess(review)
    if lstm:
        review_seq =(vectorizer.transform([review]).toarray(), maxlen=max_len)
        return model.predict(review_seq)[0]
    else:
        review_tfidf = vectorizer.transform([review]).toarray()
        return model.predict(review_tfidf)[0]

new_review = "The movie was great and I enjoyed it."
print("Logistic Regression Sentiment:", predict_sentiment(new_review, logistic_model, vectorizer))
print("LSTM Sentiment:", predict_sentiment(new_review, lstm_model, vectorizer, lstm=True))

# Step 8: End Project
