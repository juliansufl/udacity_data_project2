# ETL Pipeline

## In this code is the ETL process create for load and clean the data following in extract, transform, clean and save

### 1. Import the libraries need to the process

import requests
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

## 2.Extract data
## This process load the data and after merges all information need it
### load data
def load_data(messages_filepath, categories_filepath):
    '''
    INPUT: 
    
    messages_filepath = database that containg information of messages
    categories_filepath = database that containg information of the features

    OUTPUT:

    df = database that containg the merge of messages_filepath and categories_filepath
    '''
    #load databases
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merge data
    df=messages.merge(categories, on='id')

    return df

## 3. Clean data
## This process clean the principal df that would be use

def clean_data(df):
    '''
    This function clean the data base
    
    The database categories need to split, the information need to be separate by ';'

    Información need to be define a name in the columns and a binary values
    
    Remove dupliates

    INPUT
    df = data to clean
    OUTPUT
    df =  clean with new information need it
    '''
    #3.1
    #create the new columns using a split method
    categories = df.categories.str.split(';',expand=True)
    #bring the names that every columns would have
    row = categories.iloc[0,:]
    #remove de values and just left the name
    category_colnames = row.apply(lambda x : x[:-2])
    #remane the columns of categories
    categories.columns = category_colnames


    #3.2
    #Iterate through the category in the database for leave the binary value
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    #3.3
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    #3.4
    # drop duplicates and fix binaries results
    df.drop_duplicates(inplace=True)
    df['related'] = df['related'].map(lambda x : 1 if x == 2 else x)
    df.drop('child_alone',axis=1,inplace=True)


    return df

# 4.Save the information

def save_data(df, database_filename):
    '''
    This function save the procces above and create the db SQL

    INPUT: database clean

    OUTPUT: sql database
    '''

    engine = create_engine('sqlite:///'+database_filename)
    table_name = 'DisasterResponse'
    df.to_sql(table_name, engine, index=False,if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()