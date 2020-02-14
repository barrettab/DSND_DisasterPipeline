# Disaster Response Pipeline Project Project For Udacity DSND 
This project is part of Udacity's Data Science Nanodegree program.  The goal of this project is to deonstrate use of pipelines and NLP as well as deploying as webapp.  

# Requirements: 
- ptyhon 3.x
- pandas
- numpy
- re
- pickle
- nltk
- sklearn
- sqlchemy
- flask
SQLAlchemy
- plotly

# Repo File Structure 
- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model (note this is note uploaded to being a very large file -- to create this on your own just follow the instructions and run the train_classifier.py)

- README.md



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. Enter in text to local hosted webapp 





This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. Please refer to Udacity Terms of Service for further information.

