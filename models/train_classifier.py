from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def load_data(database):
    '''
    input:
        database: filepath database is saved
    output:
        X: training list
        Y: training targets
        categories: category names of training targets
    '''
    engine=create_engine('sqlite:///' + database)
    disaster_df = pd.read_sql_table('df', engine)
    categories=list(disaster_df.columns[4:])
    X=disaster_df['message'].values
    Y=disaster_df[categories].values
    return X,Y,categories

def tokenize(text):
    '''
    input:
        text: message text
    output:
        token_list: list of tokenized words from message
    '''
    text=re.sub(r"[^a-zA-Z0-9]", ' ', text)
    tokens=word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer= WordNetLemmatizer()
    token_list = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return token_list

def build_model():
    '''
    Build a pipeline using GridSearchCV
    '''
    pipeline=Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__min_samples_split': [2,3],
        'clf__estimator__n_estimators': [50, 100],
        'vect__ngram_range': ((1, 1),(1,2))
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv

def evaluate_model(pipeline, X_test, Y_test, categories):
    '''
    use classification report to evaluate the accuracy of the model
    '''
    Y_pred = pipeline.predict(X_test)

    result = classification_report(Y_test, Y_pred, target_names=categories)
    return result

def save_model(pipeline, pickle_filepath):
    '''
    save model to pickle file
    '''
    print('save model')
    pickle.dump(pipeline, open(pickle_filepath,'wb'))


def main():

    if len (sys.argv)==3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data from {database_filepath}")

        X, y, categories=load_data(database_filepath)

        X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2)

        print("Building model...")

        model=build_model()

        print('Training model...')

        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, categories)

        print(f'Saving model to classifier.pkl')
        save_model(model, 'classifier.pkl')

        print('trained model saved')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

        

if __name__=='__main__':
    main()
