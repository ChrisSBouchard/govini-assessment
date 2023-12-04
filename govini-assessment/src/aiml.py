import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import joblib

def load_process_set(data_type="train"):
    path = f"{os.path.dirname(os.path.realpath(__file__))}/../datasets/split/{data_type}.parquet"
    df = pd.read_parquet(path, engine='fastparquet')
    df['text'] = df.apply(lambda row: f"{row['title']} {row['abstract']} {row['code']} {row['cpc_first_4']}", axis=1)
    return df[['text', 'labels']]


def train():
   df = load_process_set().drop_duplicates()
   vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english', min_df=20)
   X = vectorizer.fit_transform(df['text'])
   y = df['labels']
   model = LogisticRegression(random_state=42)
   model.fit(X, y)
   path = f"{os.path.dirname(os.path.realpath(__file__))}/../models/"
   joblib.dump(model, f"{path}logreg.pkl")
   joblib.dump(vectorizer, f"{path}vectorizer.pkl")

def test():
    df = load_process_set(data_type="test")
    path = f"{os.path.dirname(os.path.realpath(__file__))}/../models/"
    clf = joblib.load(f"{path}logreg.pkl")
    vectorizer = joblib.load(f"{path}vectorizer.pkl")

    pred = clf.predict(vectorizer.transform(df['text']))

    print(classification_report(df['labels'].values, pred))

    
