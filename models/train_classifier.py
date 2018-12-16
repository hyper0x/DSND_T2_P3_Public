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


def build_model():
    # pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf',
    #                       MultiOutputClassifier(
    #                           RandomForestClassifier(random_state=42)))])

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', BinaryRelevance(GaussianNB()))])

    parameters = {
        'vect__max_df': (0.5, 1.0),
        'tfidf__smooth_idf': (True, False)
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        y_test, y_pred = Y_test[:, i], Y_pred[:, i]
        print(classification_report(y_test, y_pred))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start_time = time.process_time()
        model.fit(X_train, Y_train)
        end_time = time.process_time()
        print('Training time: {}s'.format(end_time - start_time))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print("Best Parameters:", model.best_params_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()