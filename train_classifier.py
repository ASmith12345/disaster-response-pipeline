from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier

def read_data(database):
    engine=create_engine('sqlite:///' + database)
    disaster_df = pd.read_sql_table('df', engine)
    categories=disaster_df.columns[4:]
    X=disaster_df['message'].values
    Y=disaster_df[categories].values
    return X,Y,categories

def tokenize(text):
    text=re.sub(r"[^a-zA-Z0-9]", ' ', text)
    tokens=word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer= WordNetLemmatizer()
    token_list = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return token_list

def build_model():
    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

            ])),

            ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators = 100)))
        ])



    return pipeline

def evaluate_model(pipeline, X_test, Y_test, categories):
    print('evaluate model')
    Y_pred = pipeline.predict(X_test)

    for col in Y_test.columns:
        print(f"{col}: {classification_report(Y_test[col], Y_pred[col], target_names=categories)}")

def save_model(pipeline, pickle_filepath):
    print('save model')
    pickle.dump(pipeline, open(pickle_filepath,'wb'))


def main():

    if len (sys.argv)==3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"loading data from {database_filepath}")

        X, y, categories=read_data(database_filepath)
        print('error reading database')

        X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2)
        print('error splitting model')

        print("building model")

        model=build_model()
        print('error building model')

        print('training model')

        model.fit(X_train, y_train)

        print('evaluating model')
        evaluate_model(model, X_test, y_test, categories)

        print(f'saving model to {model_filepath}')
        save_model(model, model_filepath)

        print('trained model saved')
    else:
        print('please provide the database name and the name of the pickle file')
        

if __name__=='__main__':
    main()
