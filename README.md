# DSND_T2_P3
The project for P3 in Term 2 of DSND.

## Disaster Response Pipeline Project

## Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Licensing](#licensing)

### Installation <a name="installation"></a>

The code in this project is written in the Python, and the specific version is `3.7.*`.

The libraries required to run all the code in this project have been documented in the file named requirements.txt.

### Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing <a name="licensing"></a>

See [LICENSE](LICENSE) for details.