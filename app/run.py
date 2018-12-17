import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys
sys.path.append('../')
from models.train_classifier import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # show distribution of different genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # show distribution of different category
    categories = list(df.columns[4:])
    category_counts = []
    for column_name in categories:
        category_counts.append(np.sum(df[column_name]))

    # show distribution of different category group by genre
    traces = []
    for genre in genre_names:
        df_genre = df[df['genre'] == genre]
        category_counts = []
        for column_name in categories:
            category_counts.append(np.sum(df_genre[column_name]))

        trace = Bar(x=categories, y=category_counts, name=genre)
        traces.append(trace)

    # show top N message categories
    category_vals = df.iloc[:, 4:]
    category_means = category_vals.mean().sort_values(ascending=False)[0:15]
    category_names = list(category_means.index)

    # create visuals
    graphs = \
    [{
        'data': [Bar(x=genre_names, y=genre_counts)],
        'layout': {
            'title': 'Distribution of Message Genres',
            'xaxis': {
                'title': "Genre"
            },
            'yaxis': {
                'title': "Count"
            }
        }
    },
    {
        'data': [Bar(x=categories, y=category_counts)],
        'layout': {
            'title': 'Distribution of Message Categories',
            'xaxis': {
                'title': "Category",
                'showticklabels': True,
                'tickangle': -30
            },
            'yaxis': {
                'title': "Count"
            }
        }
    },
    {
        'data': traces,
        'layout': {
            'title': 'Distribution of Message Categories Group by Genres',
            'barmode': 'group',
            'xaxis': {
                'title': "Category",
                'showticklabels': True,
                'tickangle': -30
            },
            'yaxis': {
                'title': "Count"
            }
        }
    },
    {
        'data': [Bar(x=category_names, y=category_means)],
        'layout': {
            'title': 'Top 15 Message Categories',
            'xaxis': {
                'title': "Categories",
                'showticklabels': True,
                'tickangle': -30
            },
            'yaxis': {
                'title': "Ratio"
            }
        }
    }]

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
        'go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()