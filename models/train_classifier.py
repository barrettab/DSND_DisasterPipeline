import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt','stopwords','wordnet'])

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score
from sklearn.model_selection  import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    '''
    Function to load data from sql database as dataframe
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message'].astype(str)
    y = df.iloc[:, 4:].astype(str)
    category_names = list(df.columns[4:])
    return X,y,category_names


def tokenize(text):
    """
    Function to tokenize the text for training model 
    """
    #normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower()) 
    #token messages
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]  
    #lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    
def build_model():
    """
    Function to build model 
    """  
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=60,random_state=20)))
    ])  
    parameters = {
        'clf__estimator__criterion':['entropy']
    }   
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv

def evaluate_model(model, y_test, y_pred, category_names,X_test):
    """
    Prints classification report 
    """

    # Generate predictions
    #def evaluate_model(y_test, y_pred,category_names):
    
    y_preds = model.predict(X_test)
    for label, pred, col in zip(y_test.values.transpose(), y_preds.transpose(), 
                                category_names):
        print(col)
        print(classification_report(label, pred)) 

def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    '''
    pickle.dump(model, open(model_filepath, "wb"))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
      
        print('Training model...')
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        
        print('Evaluating model...')
        evaluate_model(model, y_test, y_pred, category_names,X_test)

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
