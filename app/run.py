import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
# model = joblib.load("../models/classifier.pkl")
model = joblib.load("../classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    weather_related_counts = df.groupby('weather_related').count()['message']
    weather_related_names = list(weather_related_counts.index)

    aid_related_counts = df.groupby('aid_related').count()['message']
    aid_related_names = list(aid_related_counts.index)

    infrastructure_related_counts = df.groupby('infrastructure_related').count()['message']
    infrastructure_related_names = list(infrastructure_related_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=weather_related_names,
                    y=weather_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Weather Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "weather related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=aid_related_names,
                    y=aid_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Aid Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "aid related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=infrastructure_related_names,
                    y=infrastructure_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Infrastructure Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "infrastructure related"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)
    print('running run.py')


if __name__ == '__main__':
    main()