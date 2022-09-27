# udacity_data_project2

# Disaster Response Pipeline


# 1. Motivation

The motivation of this project is create a webapp that have a train model using NPL that can help to predict what it is the most important needs that people could have when a natural disaster happen, this can help that the emergencies supports have all supplies in desk and the resources get correct to everyone that report what it is the more needs that the people have.

# 2. File Description

This project have three data resources that are located in the data folder:

* disaster_categories: it is a csv that has the categories that the model predict
* disaster_messages: it is a csv that has the message that people write, this information helps to train the model and predict new categories
* DisasterResponde: it is a DB file that it is create after the ETL pipeline that clean the two csv files that left ready for procces in the ML pipeline

# 3. Instructions

This github has three folders: 

* App: this folder have the run.py that make the webapp to run and also the temples of the web site.
* Data: it has the csv's file, it also have the py to run the clean process and a ipynb that have the process not like a process
* Models: it has the ML py process for the model and the ML ipynb not like a process

For run this process you need to run first process_data.py that it is on data, after train_classifier.py it is on models and the run.py that it is on app

# 4. Requirements

For run this process you need python 3 or newest and packages as:

* Numpy
* Pandas
* Skelearn
* NLTK - NPL
* SQLlite and SQLalchemy
* re
* Pickle

Other libraries that you can find in this process

# 5. Acknowledgments

scikit-learn developers. (n.d.). sklearn.ensemble.RandomForestClassifier — scikit-learn 1.1.1 documentation. Retrieved Sep 23, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Pedregosa, F., Weiss, R., & Brucher, M. (2011). Scikit-learn : Machine Learning in Python. 12, 2825–2830.

# Thank you! Gracias totales!
