# ML Pipeline

## In this code is the ML process create for genate a model

### 1. Import the libraries need to the process

from copyreg import pickle
import sys
import requests
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import os
import re
import string
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,classification_report
from sklearn.multioutput import MultiOutputClassifier

## 2. load data for modeling
def load_data(database_filepath):
    '''
    This function load and transform the data for future use for modeling

    INPUT: df from a db format that was create in process_data.py

    OUTPUT:
    X: the independent variables
    Y: Target
    columns_name: the columns' names of the Y variables
    '''
    #load the data
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'DisasterResponse'
    df = pd.read_sql_table(table_name, engine)

    #create the outputs
    X = df['message']
    y = df.iloc[:,4:]
    columns_names = y.columns

    return X, y, columns_names

## 3. generated the tokenize

def tokenize(text):
    '''
    This process clean the message by tokenize and lemmatize
    
    INPUT: df with the message

    OUTPUT: df modified 
    '''
    # this part handle some part of the special characters

    text= re.sub(r"[^a-zA-Z0-9]"," ",text)

    # this part tokenize and remove stopwords and other class of special characters
    text_tokenize = word_tokenize(text)
    text_tokenize = [y for y in text_tokenize if y not in stopwords.words('english')+list(string.punctuation)]

    #define to lemmatize
    lemmatizer=WordNetLemmatizer()

    #This lemmatize the input
    clean_tokens = []
    for tok in text_tokenize:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    #get the clean information
    return clean_tokens

## 4. Define model

def build_model():
    '''
    This function create a pipeline for the model
    INPUT: NONE

    OUTPUT: pipeline of the model
    
    '''
    #establish some parameters for improve del model
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    parameters  = { "clf__estimator__n_estimators": [50,100,150], 
                "clf__estimator__max_depth" : [1,2,3,4,5,6],
                "clf__estimator__max_samples": [0.05, 0.1],
                'clf__estimator__random_state':[42]}
    cv = pipeline
    
    return cv


#################################################################
## Process with GridSearchCV can take more that for that reason ##
## it is not use, if you want to just take out the comment      ##
#################################################################
#def build_model():
#    '''
#    This function create a pipeline for the model
#    INPUT: NONE

#    OUTPUT: pipeline of the model
    
#    '''
    #establish some parameters for improve del model
#    pipeline = Pipeline([
#        ('features', FeatureUnion([

#            ('text_pipeline', Pipeline([
#                ('vect', CountVectorizer(tokenizer=tokenize)),
#                ('tfidf', TfidfTransformer())
#            ])),
#        ])),

#        ('clf', MultiOutputClassifier(RandomForestClassifier()))
#    ])

#    parameters  = { "clf__estimator__n_estimators": [50,100,150], 
#                "clf__estimator__max_depth" : [1,2,3,4,5,6],
#                "clf__estimator__max_samples": [0.05, 0.1],
#                'clf__estimator__random_state':[42]}

#    cv = GridSearchCV(pipeline,param_grid=parameters)

#    return cv

## 5. Evalute model

def evaluate_model(model, X_test, Y_test, columns_names):
    '''
    This function generate the best model to implement and create the prediction and give the metrics of data model
    INPUT: model, test bases and columns of names of y
    OUTPUT: NONE
    '''
    #prediction
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)

    #calculation and evaluation of the model
    accuracy_mean = []
    f1_mean = []
    precision_mean = []
    recall_mean = []

    for i in range(0,len(columns_names.shape),1):
        accuracy_mean.append(accuracy_score(Y_test.iloc[:,i], y_pred.iloc[:,i]))
        f1_mean.append(f1_score(Y_test.iloc[:,i], y_pred.iloc[:,i],average='macro'))
        precision_mean.append(precision_score(Y_test.iloc[:,i], y_pred.iloc[:,i],average='macro'))
        recall_mean.append(recall_score(Y_test.iloc[:,i], y_pred.iloc[:,i],average='macro'))

    accuracy_mean = pd.DataFrame(accuracy_mean)
    f1_mean = pd.DataFrame(f1_mean)
    precision_mean = pd.DataFrame(precision_mean)
    recall_mean = pd.DataFrame(recall_mean)

    print('The mean accuracy score of the model is',accuracy_mean.mean())
    print('The mean precision score of the model is',precision_mean.mean())
    print('The mean recall of the model is',recall_mean.mean())
    print('The mean F1 score of the model is',f1_mean.mean())

# 6. save model

def save_model(model, model_filepath):
    '''
    This function save the train model
    
    INPUT: Model fitted and go to filepath

    OUTPUT: NONE

    '''
    pickle.dump(model,open(model_filepath,'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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