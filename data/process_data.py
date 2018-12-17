import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data from the specified CSV files.

    Parameters
    ----------
    messages_filepath : str
        The path to the CSV file containing the message data.

    categories_filepath : str
        The path to the CSV file containing the category data.

    Returns
    -------
    df : pandas.Dataframe
        The dataframe containing message data and category data.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='outer', on='id')

    return df


def clean_data(df):
    """Transform and clean up data.

    Parameters
    ----------
    df : pandas.Dataframe
        The dataframe containing message data and category data.

    Returns
    -------
    df : pandas.Dataframe
        The dataframe containing clean data.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        # also, convert the character to integer
        categories[column] = categories[column].apply(
            lambda x: 1 if int(x[-1]) > 0 else 0)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename, table_name):
    """Store data in the database.

    Parameters
    ----------
    df : pandas.Dataframe
        The dataframe containing data.

    database_filename : str
        The path to the database file.

    table_name : str
        The name of table.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, if_exists='replace', index=False)


def main():
    """The main function.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(
            messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        table_name = 'Messages'
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


def check():
    """Check the validity of the database file.
    """
    if len(sys.argv) == 4:
        table_name = 'Messages'
        database_filepath = sys.argv[3]
        print('Check the database located at \'{}\'...'.format(
            database_filepath))
        print('Query the count of table \'{}\'...'.format(table_name))
        sql = 'SELECT COUNT(*) FROM {}'.format(table_name)
        engine = create_engine('sqlite:///' + database_filepath)
        print(pd.read_sql(sql, engine))


if __name__ == '__main__':
    main()
    check()