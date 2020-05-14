import pandas as pd
import numpy as np
from scipy.stats import mode
import string
from functions.utilities import *
import seaborn as sns  

#read the data
trainpath = './data/train.csv'
testpath = './data/test.csv'
traindf = pd.read_csv(trainpath, delimiter=",")
testdf = pd.read_csv(testpath, delimiter=",")
fulldf = traindf.merge(testdf, how="outer")

dfs = [fulldf, testdf, traindf]

def nbEmbarked_missingNA(df,embarked):
    return df.loc[(df['Embarked'] == embarked) & (df['Sex'] == 'female') & (df['Pclass'] == 1)].shape[0]

nbEmbarked_missingNA = list(map(lambda x: nbEmbarked_missingNA(fulldf, x), ['S', 'C', 'Q']))
def processEmbarked(df):
    df['Embarked'] = np.where((df['Embarked'].isna()), 'C', df['Embarked'])
    
updateAllDataSets(processEmbarked, dfs)

title_dict = {
    "Mrs": "Mrs",
    "Ms": "Mrs",
    "Mme": "Mrs",
    "Miss": "Miss",
    "Mlle": "Miss",
    "Mr": "Mr",
    "Master": "Master",
    "Major": "Officer",
    "Col": "Officer",
    "Capt": "Officer",
    "Rev": "Officer",
    "Dr": "Officer",
    "Countess": "Royalty",
    "Sir": "Royalty",
    "Lady": "Royalty",
    "Don": "Royalty",
    "Jonkheer": "Royalty"
}

def getTitles(df, dict, fulldf):
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, dict.keys()))
    data_type_dict['Title'] = 'nominal'
    df['Title'] = df['Title'].map(dict)
    
updateAllDataSets(getTitles, dfs, title_dict, fulldf)

#Turning cabin number into Deck
deck_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown'] #T is removed as only used once

def addDeck(df, deck_list):
    df.Cabin = df.Cabin.fillna('Unknown')
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, deck_list))

updateAllDataSets(addDeck, dfs, deck_list)

def fill_age(row, grouped):
    condition = (
        (grouped['Sex'] == row['Sex']) & 
        (grouped['Title'] == row['Title']) & 
        (grouped['Pclass'] == row['Pclass'])
    )
    return grouped[condition]['Age'].values[0]

def process_age(df, grouped):
    df['Age'] = df.apply(lambda row: fill_age(row, grouped) if np.isnan(row['Age']) else row['Age'], axis=1)
    
updateAllDataSets(process_age, dfs, grouped_median)
age_bin_edges = [0,
                 10,
                 20,
                 30,
                 40,
                 50,
                 60,
                 70,
                 max(fulldf['Age'])+1]

def addAgeBins(df, age_bin_edges):
    df['Age_bins'] = pd.cut(df['Age'], 
                             age_bin_edges, 
                             labels=[1,2,3,4,5,6,7,8])
    
updateAllDataSets(addAgeBins, dfs, age_bin_edges)

def processFare(df, category, categories, fulldf):
    for category in categories :
        df['Fare'] = np.where(((df['Fare'].isna()) & (df['Pclass'] == category)), 
                              [fulldf.loc[fulldf['Pclass'] == category].loc[:,"Fare"].median()] , 
                              df['Fare'])
        df['Fare'] = np.where(((df['Fare'] == 0) & (df['Pclass'] == category)), 
                              [fulldf.loc[fulldf['Pclass'] == category].loc[:,"Fare"].median()] , 
                              df['Fare'])
updateAllDataSets(processFare, dfs, category, categories, fulldf)

fare_bin_edges = [0,
                  fulldf['Fare'].quantile(0.2),
                  fulldf['Fare'].quantile(0.4),
                  fulldf['Fare'].quantile(0.6),
                  fulldf['Fare'].quantile(0.8),
                  fulldf['Fare'].quantile(0.9),
                  max(fulldf['Fare'])+1]

def addFareBins(df, fare_bin_edges):
    df['Fare_bins'] = pd.cut(df['Fare'], 
                             fare_bin_edges, 
                             labels=[1,2,3,4,5,6])
    
updateAllDataSets(addFareBins, dfs, fare_bin_edges)

def processFamily(df):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    
    df['NoFamily'] = df['FamilySize'].map(lambda x: 1 if x == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda x: 1 if 2 <= x <= 4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda x: 1 if 5 <= x else 0)
    
updateAllDataSets(processFamily, dfs)

def dummies(df):
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_dummies], axis=1)
    
    deck_dummies = pd.get_dummies(df['Deck'], prefix='Deck')
    df = pd.concat([df, deck_dummies], axis=1)
    
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    
    sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
    df = pd.concat([df, sex_dummies], axis=1)
    
    return df    
    
def drop_features(df):
    df.drop(axis=1, 
            labels=['Name', 'Cabin', 'Fare', 'FamilySize', 'Deck', 'Deck_Unknown', 'Embarked', 'Age', 'Sex', 'Title',  'Ticket'], 
            inplace=True)

testdf = dummies(testdf)
traindf = dummies(traindf)
fulldf = dummies(fulldf)

dfs = [fulldf, testdf, traindf]
updateAllDataSets(drop_features, dfs)

testdf.to_csv("testdf-2.csv", index=False)
traindf.to_csv("traindf-2.csv",index=False)