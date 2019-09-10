import os 
import csv 
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf

import numpy as np
import pandas as pd
import matplotlib as pt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def model(test_data):



    if "__name__" == "__main__":
        dtypes = {
        's.no':'int64',
        'Timestamp':'object',
        'Age' : 'float64',
        'Gender' : 'category',
        'Country' : 'category',
        'state' : 'category',
        'self_employed' : 'category',
        'family_history' : 'category',
        'treatment' : 'category',
        'work_interfere' : 'category',
        'no_employees' : 'category',
        'remote_work' : 'category',
        'tech_company' : 'category',
        'benefits' : 'category',
        'care_options' : 'category',
        'wellness_program' : 'category',
        'seek_help' : 'category',
        'anonymity' : 'category',
        'leave' : 'category',
        'mental_health_consequence' : 'category',
        'phys_health_consequence' : 'category',
        'coworkers' : 'category',
        'supervisor' : 'category',
        'mental_health_interview' : 'category',
        'phys_health_interview' : 'category',
        'mental_vs_physical' : 'category',
        'obs_consequence' : 'category',
        'comments' : 'str'    
    }

    #import Data
    train_df = pd.read_csv('trainms.csv',dtype = dtypes)
    test_df = test_data
    samdf = pd.read_csv('samplems.csv', dtype=dtypes)
    test_df['treatment'] = samdf['treatment']
    #Data preprocessing

    train_df['state'] = train_df['state'].fillna('CA')
    train_df['self_employed'] = train_df['self_employed'].fillna('No')
    train_df['work_interfere'] = train_df['work_interfere'].fillna('Sometimes')

    frames = [train_df, test_df]
    result = pd.concat(frames)

    #Male
    result.loc[result['Gender'] == 'M','Gender'] = 'Male'
    result.loc[result['Gender'] == 'male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'malr','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Malr','Gender'] = 'Male'
    result.loc[result['Gender'] == 'mail','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Male ','Gender'] = 'Male'
    result.loc[result['Gender'] == 'msle','Gender'] = 'Male'
    result.loc[result['Gender'] == 'm','Gender'] = 'Male'
    result.loc[result['Gender'] == 'maile','Gender'] = 'Male'
    result.loc[result['Gender'] == 'mal','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Mal','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Male-ish','Gender'] = 'Male'
    result.loc[result['Gender'] == 'ostensibly male, unsure what that really means','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Cis Man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'something kinda male?','Gender'] = 'Male'
    result.loc[result['Gender'] == 'make','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Make','Gender'] = 'Male'
    result.loc[result['Gender'] == 'cis Man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Cis Male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'cis Male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'cis male','Gender'] = 'Male'
    result.loc[result['Gender'] == 'Man','Gender'] = 'Male'
    result.loc[result['Gender'] == 'man','Gender'] = 'Male'



    #Female
    result.loc[result['Gender'] == 'F','Gender'] = 'Female'
    result.loc[result['Gender'] == 'female','Gender'] = 'Female'
    result.loc[result['Gender'] == 'femail','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Female ','Gender'] = 'Female'
    result.loc[result['Gender'] == 'f','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Cis Female','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Femake','Gender'] = 'Female'
    result.loc[result['Gender'] == 'cis-female/femme','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Female (cis)','Gender'] = 'Female'
    result.loc[result['Gender'] == 'cis female','Gender'] = 'Female'
    result.loc[result['Gender'] == 'Woman','Gender'] = 'Female'
    result.loc[result['Gender'] == 'woman','Gender'] = 'Female'

    #Transgender
    result.loc[result['Gender'] == 'Trans woman','Gender'] = 'Transgender'
    result.loc[result['Gender'] == 'Female (trans)','Gender'] = 'Transgender'
    result.loc[result['Gender'] == 'Female (trans)','Gender'] = 'Transgender'
    result.loc[result['Gender'] == 'Trans-female','Gender'] = 'Transgender'


    #Others
    result.loc[result['Gender'] == 'non-binary','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Nah','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Enby','Gender'] = 'Others'
    result.loc[result['Gender'] == 'fluid','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Genderqueer','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Androgyne','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Agender','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Guy (-ish) ^_^','Gender'] = 'Others'
    result.loc[result['Gender'] == 'male leaning androgynous','Gender'] = 'Others'
    result.loc[result['Gender'] == 'Neuter','Gender'] = 'Others'
    result.loc[result['Gender'] == 'queer','Gender'] = 'Others'
    result.loc[result['Gender'] == 'A little about you','Gender'] = 'Others'
    result.loc[result['Gender'] == 'p','Gender'] = 'Others'


    df_sex = pd.get_dummies(result['Gender'])
    df_new = pd.concat([result, df_sex], axis=1)
    result['Male'] = df_new['Male']
    result['Female'] = df_new['Female']
    result['Transgender'] = df_new['Transgender']
    result['Others'] = df_new['Others']

    #data combining
    data = result.loc[:,['s.no','Age','Male','Female','Transgender','Others']]

    data['Country'] = pd.factorize(result['Country'], sort=True)[0]
    #1 - change gender with dummy variables--> male,female,Transgender,Others
    data['Gender'] = pd.factorize(result['Gender'], sort=True)[0]

    data['Timestamp'] = pd.factorize(result['Timestamp'], sort=True)[0]
    data['anonymity'] = pd.factorize(result['anonymity'], sort=True)[0]
    data['benefits'] = pd.factorize(result['benefits'], sort=True)[0]
    data['care_options'] = pd.factorize(result['care_options'], sort=True)[0]
    data['comments'] = pd.factorize(result['comments'], sort=True)[0]
    data['coworkers'] = pd.factorize(result['coworkers'], sort=True)[0]
    data['family_history'] = pd.factorize(result['family_history'], sort=True)[0]
    data['leave'] = pd.factorize(result['leave'], sort=True)[0]
    data['mental_health_consequence'] = pd.factorize(result['mental_health_consequence'], sort=True)[0]
    data['mental_health_interview'] = pd.factorize(result['mental_health_interview'], sort=True)[0]
    data['mental_vs_physical'] = pd.factorize(result['mental_vs_physical'], sort=True)[0]
    data['no_employees'] = pd.factorize(result['no_employees'], sort=True)[0]
    data['obs_consequence'] = pd.factorize(result['obs_consequence'], sort=True)[0]
    data['phys_health_consequence'] = pd.factorize(result['phys_health_consequence'], sort=True)[0]
    data['phys_health_interview'] = pd.factorize(result['phys_health_interview'], sort=True)[0]
    data['remote_work'] = pd.factorize(result['remote_work'], sort=True)[0]
    data['seek_help'] = pd.factorize(result['seek_help'], sort=True)[0]
    data['self_employed'] = pd.factorize(result['self_employed'], sort=True)[0]
    data['state'] = pd.factorize(result['state'], sort=True)[0]
    data['supervisor'] = pd.factorize(result['supervisor'], sort=True)[0]
    data['tech_company'] = pd.factorize(result['tech_company'], sort=True)[0]
    data['treatment'] = pd.factorize(result['treatment'], sort=True)[0]
    data['wellness_program'] = pd.factorize(result['wellness_program'], sort=True)[0]
    data['work_interfere'] = pd.factorize(result['work_interfere'], sort=True)[0]


    treatment = data['treatment']
    y = treatment                    #treatment is the predicting class

    age = data['Age']
    country = data['Country']

    #1 - newly added dummy variable

    Male = data['Male']
    Female = data['Female']
    Transgender = data['Transgender']
    Others = data['Others']pd.factorize(

    #--------------------------

    anonymity = data['anonymity'] 
    benefits = data['benefits']
    care_options = data['care_options']
    comments = data['comments'] 
    coworkers = data['coworkers']
    family_history = data['family_history'] 
    leave = data['leave'] 
    mental_health_consequence = data['mental_health_consequence']
    mental_vs_physical = data['mental_vs_physical']
    obs_consequence = data['obs_consequence'] 
    phys_health_consequence = data['phys_health_consequence'] 
    seek_help = data['seek_help'] 
    state = data['state'] 
    supervisor = data['supervisor'] 
    wellness_program = data['wellness_program'] 
    work_interfere = data['work_interfere']


    x = np.column_stack((age,country,Male,Female,Transgender,Others,anonymity,benefits,care_options,family_history,leave,mental_health_consequence,mental_vs_physical,obs_consequence,phys_health_consequence,seek_help,seek_help,supervisor,state,wellness_program,work_interfere))


    x1_train = x[:len(train_df)]
    x1_test = x[len(train_df):]
    y1_train = y[:len(train_df)]
    y1_test = y[len(train_df):]



    logreg = LogisticRegression().fit(x1_train,y1_train)

    y_pred = logreg.predict(x1_test)


    if y_pred[0] == 1:
        return 1
    else:
        return 0






