import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    X = df['message'].values
    category_names = [
        i for i in df.columns
        if i not in ['id', 'message', 'original', 'genre']
    ]
    Y = df[category_names].values

    return X, Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    clean_tokens = [
        lemmatizer.lemmatize(tok).lower().strip() for tok in tokens
        if tok not in stop_words
    ]

    return clean_tokens


def build_pipeline():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf',
                          MultiOutputClassifier(
                              RandomForestClassifier(random_state=42)))])

    return pipeline


def build_grid_search(pipeline):
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__smooth_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, name in enumerate(category_names):
        print('{}:'.format(name))
        y_test, y_pred = Y_test[:, i], Y_pred[:, i]
        print(classification_report(y_test, y_pred))


def save_model(model, model_filepath):
    return joblib.dump(model, model_filepath, compress=('gzip', 6), protocol=4)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        pipeline = build_pipeline()
        gscv = build_grid_search(pipeline)

        print('Training model...')
        start_time = time.time()
        gscv.fit(X_train, Y_train)
        end_time = time.time()
        print('Training time: {}s'.format(end_time - start_time))

        print("Best score:", gscv.best_score_)
        print("Best parameters:", gscv.best_params_)

        best_model = gscv.best_estimator_
        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        filenames = save_model(best_model, model_filepath)

        print('Trained model saved! (filenames: {})'.format(filenames))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


def check():
    if len(sys.argv) == 3:
        model_filepath = sys.argv[2]
        print('Check the model located at \'{}\'...'.format(model_filepath))
        joblib.load(model_filepath)


if __name__ == '__main__':
   # main()
    check()
