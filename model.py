#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import cython
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, classification_report
from scipy import cluster
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
pd.options.mode.chained_assignment = None


# In[2]:


# Name our columns after analysis data-sheet from dataset source

names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Neuroticism', 'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness', 'Sensation_seeking', 'Alcohol', 'Amphetamine', 'Amyl_nitrite', 'Benzodiazepine', 'Caffeine', 'Cannabis', 'Chocolate', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legal_highs', 'LSD', 'Methadone', 'Mushrooms', 'Nicotine', 'Semeron', 'VSA']


# In[3]:


data = pd.read_csv('drug_consumption.data', header = None, names = names)


# In[4]:


# Observe top 10 observations

data.head(10)


# In[5]:


# Observe last 10 observations

data.tail(10)


# In[6]:


data.shape


# In[7]:


# Count number of NaN's in every column

print(data.isna().sum())


# In[8]:


# Create a variable containing all the columns/features names

data_columns = data.columns

print(data_columns)


# In[9]:


# Count number of unique values in every column

data_nunique_dict = data.nunique().to_dict()
data_nunique_dict


# In[10]:


# Display basic data statistics

data.describe()


# In[11]:


# set data index

data.set_index('ID', inplace = True)


# In[12]:


data.head()


# In[13]:


# Creating rough version of classification of drug consumption. Modifing my existing `data` object:
# 1 - if a person used a drug in month, week or day, then let's say that he did consume a drug.

# 0 - other categories are placed into the group that he did not consume a drug; 

def change(a):
    
    if ((a == 'CL6') or (a == 'CL5') or (a == 'CL4') ):
        a = 1
    
    elif ((a == 'CL0') or (a == 'CL1') or (a == 'CL2') or (a == 'CL3')):
        a = 0
    
    return a


# In[14]:


# Applying our changes in classification of drug consumption to columns with drugs

data['Amphetamine'] = data['Amphetamine'].map(change)

data['Amyl_nitrite'] = data['Amyl_nitrite'].map(change)

data['Benzodiazepine'] = data['Benzodiazepine'].map(change)

data['Cannabis'] = data['Cannabis'].map(change)

data['Cocaine'] = data['Cocaine'].map(change)

data['Crack'] = data['Crack'].map(change)

data['Ecstasy'] = data['Ecstasy'].map(change)

data['Heroin'] = data['Heroin'].map(change)

data['Ketamine'] = data['Ketamine'].map(change)

data['LSD'] = data['LSD'].map(change)

data['Methadone'] = data['Methadone'].map(change)

data['Mushrooms'] = data['Mushrooms'].map(change)

data['Semeron'] = data['Semeron'].map(change)

data['VSA'] = data['VSA'].map(change)

data['Alcohol'] = data['Alcohol'].map(change)

data['Legal_highs'] = data['Legal_highs'].map(change)

data['Nicotine'] = data['Nicotine'].map(change)

data['Chocolate'] = data['Chocolate'].map(change)

data['Caffeine'] = data['Caffeine'].map(change)


# In[15]:


#There is a problem with float64 values in dataset. I converted values to  .f5
#Because if you see the number 0.86054, for example, then its not what it is actually.
#It can be 0.8605400000001. That's why the comparison does not work

def toFixed(x):
    x = float('{:.5f}'.format(x))
    return x

for i in list(data.columns):
    data[i] = data[i].map(toFixed)


# In[16]:


#Decode column Age 

#'18-24' age -> 0
#'25-34' age -> 1
#'35-44' age -> 2
#'45-54' age -> 3
#'55-64' age -> 4
#'65+'   age -> 5

def changeAge(x):
    if (x == -0.95197):
        x = 0
    elif (x == -0.07854):
        x = 1
    elif (x == 0.49788):
        x = 2
    elif (x == 1.09449):
        x = 3
    elif (x == 1.82213):
        x = 4
    elif (x == 2.59171):
        x = 5
    return x

data['Age'] = data['Age'].map(changeAge)


# In[17]:


#Decode Gender

# Female -> 0
# Male   -> 1

def changeGender(x):
    if (x == 0.48246 ):
        x = 0
    elif (x == -0.48246 ):
        x = 1
    return x

data['Gender'] = data['Gender'].map(changeGender)


# In[18]:


#Decode Education

# Left school before 16 years                          -> 0
# Left school at 16 years                              -> 1
# Left school at 17 years                              -> 2
# Left school at 18 years                              -> 3
# Some college or university, no certificate or degree -> 4
# Professional certificate/ diploma                    -> 5
# University degree                                    -> 6
# Masters degree                                       -> 7
# Doctorate degree                                     -> 8

def changeEducation(x):
  
  if (x == -2.43591):
    x = 0
  elif (x == -1.73790):
    x = 1
  elif (x == -1.43719):
    x = 2
  elif (x == -1.22751):
    x = 3
  elif (x == -0.61113):
    x = 4
  elif (x == -0.05921):
    x = 5
  elif (x == 0.45468):
    x = 6
  elif (x == 1.16365):
    x = 7
  elif (x == 1.98437):
    x = 8
  return x

data['Education'] = data['Education'].map(changeEducation)


# In[19]:


#Decode country

#Australia -> 0
#Canada    -> 1
#New Zealand->2
#Other     -> 3
#Republic of Ireland ->4
#UK         ->5
#USA        ->6

def changeCountry(x):
  
  if (x == -0.09765):
    x = 0
  elif (x == 0.24923):
    x = 1
  elif (x == -0.46841):
    x = 2
  elif (x == -0.28519):
    x = 3
  elif (x == 0.21128):
    x = 4
  elif (x == 0.96082):
    x = 5
  elif (x == -0.57009):
    x = 6
  return x

data['Country'] = data['Country'].map(changeCountry)


# In[20]:


#Decode Ethnicity

#Asian  -> 0
#Black  -> 1
#Mixed-Black/Asian -> 2
#Mixed-White/Asian -> 3
#Mixed-White/Black -> 4
#Other             -> 5
#White             -> 6

def changeEthnicity(x):
  
  if (x == -0.50212):
    x = 0
  elif (x == -1.10702):
    x = 1
  elif (x == 1.90725):
    x = 2
  elif (x == 0.12600):
    x = 3
  elif (x == -0.22166):
    x = 4
  elif (x == 0.11440):
    x = 5
  elif (x == -0.31685):
    x = 6
  return x

data['Ethnicity'] = data['Ethnicity'].map(changeEthnicity)


# In[21]:


data.tail()


# In[22]:


# Count number of unique values in every column again
# to compare whether we missed something

data_nunique_dict1 = data.nunique().to_dict()
data_nunique_dict1


# In[23]:


data["Age"] = data['Age'].astype('int')
data["Education"] = data['Education'].astype('int')
data["Country"] = data['Country'].astype('int')
data["Ethnicity"] = data['Ethnicity'].astype('int')
data["Alcohol"] = data['Alcohol'].astype('int')


# In[24]:


for i in range(12,31):
    data[data.columns[i]] = data[data.columns[i]].astype('int')


# In[25]:


import csv
with open('datav4.csv','w',newline='') as f:  #Ouverture du fichier CSV en écriture
    ecrire=csv.writer(f)                        # préparation à l'écriture
    for i in data:                           # Pour chaque ligne du tableau...  
        ecrire.writerow(i)                # Mettre dans la variable ecrire cette nouvelle ligne      
print('',end='\n')
print('longueur du tableau : ',len(data))


# In[26]:


# Observing drug consumption rate over Age

age = pd.concat([data[data['Cannabis']==1]['Age'],data[data['Cannabis']==0]['Age']],axis=1)
age.columns=['Cannabis User','Never Used Cannabis']

AgePlot = age.plot(kind='hist',bins=6,figsize=(10,6),alpha=0.3,grid=True)
AgePlot.set(ylabel = 'Number of Users', xlabel='Age')

AgeLabels = ['0','18','24','35','45','55','65+']
AgePlot.set_xticklabels(AgeLabels)


# In[27]:


# Observing drug consumption rate across Gender

sns.set(rc={'figure.figsize':(10,6)})
GenderPlot = sns.countplot(x='Cannabis',hue='Gender',data=data,palette='afmhot')

labels = ['Never Used', 'Cannabi User']
GenderPlot.set_xticklabels(labels)

GenderPlot.set(ylabel = 'Number of Users', xlabel='Cannabis Consumption')
plt.legend(title='Cannabis User', loc='upper left', labels=['Female', 'Male'])


# In[28]:


# Analyzing drug consumption rate across Education Level
sns.set(rc={'figure.figsize':(10,5)})
x = ('yes','no')
EducationPlot = sns.countplot(x='Cannabis',hue='Education',data=data,palette='rainbow')

EducationPlot.set_xticklabels(labels)
EducationPlot.set(ylabel = 'Number of Users', xlabel='Cannabis Consumption')

plt.legend(title='Education Level', loc='upper left', 
           labels=['Left School before 16', 'Left School at 16', 'Left School at 17', 'Left School at 18',
                   'Some College', 'Certificate/Diploma', 'University Degree', 'Masters', 'Doctorate'])


# In[29]:

# Analyzing drug consumption combining Age and Gender features
Age_Gender_Plot = sns.factorplot(x='Gender' , y='Age' , data=data , hue='Cannabis' , kind='violin' , palette=['g','r'] , split=True)

AgeLabels = ['0','18','24','35','45','55','65+']
genderlabels = ['Female', 'Male']
Age_Gender_Plot.set_yticklabels(AgeLabels)
Age_Gender_Plot.set_xticklabels(genderlabels)

# In[30]:


# Combine Age and Education Level together
# Focus only on the actual cannabis users

datav2 = data[data['Cannabis'] == 1]
Age_Education_Plot = sns.boxplot(x='Education',y='Age',data=datav2)

AgeLabels = ['0','18','24','35','45','55','65+']
Age_Education_Plot.set_yticklabels(AgeLabels)

EducationLabels = ['Left School before 16', 'Left School at 16', 'Left School at 17', 'Left School at 18',
                   'Some College', 'Certificate/Diploma', 'University Degree', 'Masters', 'Doctorate']
Age_Education_Plot.set_xticklabels(EducationLabels,rotation=30)

# In[31]:


from sklearn.model_selection import train_test_split

target = data["Cannabis"]

datav3 = data.drop(columns=["Cannabis","Alcohol","Amphetamine","Amyl_nitrite","Benzodiazepine","Caffeine","Cannabis","Chocolate","Cocaine","Crack","Ecstasy","Heroin","Ketamine","Legal_highs","LSD","Methadone","Mushrooms","Nicotine","Semeron","VSA"])
feature_names = datav3.columns

X_train, X_test, y_train, y_test = train_test_split(datav3, target, random_state=1, stratify=target)
X_train.head(2)


# In[32]:


# Use a heatmap quickly check if there is any NULL values within any of the features
# all shaded ==> all the cells have validate data; otherwise will be hilighted in yellow
sns.heatmap(X_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)


# In[33]:


X_train.describe()




# In[37]:




# In[38]:



# In[39]:


countries = datav2['Country'].value_counts().plot(kind='pie', figsize=(8, 8))


# In[40]:


ethnicity = datav2['Ethnicity'].value_counts().plot(kind='pie', figsize=(8, 8))


# In[41]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=False);


# In[42]:


scatter_matrix(datav3, alpha=0.2, figsize=(20, 20), diagonal='kde')
plt.show()


# In[43]:


datav4 = pd.read_csv('drug_consumption.data', header = None, names = names)
datav4['Cannabis'] = datav4['Cannabis'].map(change)
datav4['Cannabis'].value_counts()


# In[44]:


# Because sample size of individuals who have not used cannabis is significantly smaller than sample size of cannabis users
# upsampling is performed to make the sample sizes equal

from sklearn.utils import resample

data_majority = datav4[datav4['Cannabis']==0]
data_minority = datav4[datav4['Cannabis']==1]

data_minority_upsampled = resample(data_minority,
replace=True,
n_samples=1097, # same number of samples as majority classe
random_state=1) # set the seed for random resampling

# Combine resampled results
data_upsampled = pd.concat([data_majority, data_minority_upsampled])

# Assign data the data_upsampled df
datav4 = data_upsampled
datav4['Cannabis'].value_counts()


# In[45]:


# Check for nulls
datav4.isnull().sum().head()


# In[46]:


# Prepare dfs for train test split
from sklearn.model_selection import train_test_split

target = datav4["Cannabis"]
datav4 = datav4.drop(columns=["ID","Cannabis","Alcohol","Amphetamine","Amyl_nitrite","Benzodiazepine","Caffeine","Cannabis","Chocolate","Cocaine","Crack","Ecstasy","Heroin","Ketamine","Legal_highs","LSD","Methadone","Mushrooms","Nicotine","Semeron","VSA"])


datav4.head()


# In[47]:


from sklearn.preprocessing import OneHotEncoder

columnsToEncode = ['Age', 'Gender', 'Education','Country', 'Ethnicity', 'Neuroticism', 'Extraversion',
            'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness', 'Sensation_seeking']
data_reindex = datav4.reset_index(drop=True)


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

one_hot_data = one_hot(data_reindex, columnsToEncode)
one_hot_data = one_hot_data.drop(columns=['Age', 'Gender', 'Education','Country', 'Ethnicity', 'Neuroticism', 'Extraversion',
            'Openness', 'Agreeableness', 'Conscientiousness', 'Impulsiveness', 'Sensation_seeking'])
one_hot_data.head()


# In[48]:


# Train split data
X_train, X_test, y_train, y_test = train_test_split(one_hot_data, target, random_state=1, stratify=target)


# In[49]:


X_train.head()


# In[50]:


# Scale the data using Standard Scaler
from sklearn.preprocessing import StandardScaler
X_standard_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_standard_scaler.transform(X_train)
X_test_scaled = X_standard_scaler.transform(X_test)


# In[51]:


# Logistic Regression model
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(max_iter=1000000,solver='liblinear')

# Train the model
model_log.fit(X_train_scaled, y_train)

# Print scores
print(f"Training Data Score: {model_log.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {model_log.score(X_test_scaled, y_test)}")


# In[52]:


# Create the GridSearchCV model for logistic regression
from sklearn.model_selection import GridSearchCV

logistic_param_grid = {"penalty": ['l1','l2'],
              "C": [0.001,0.01,0.1,1,10,100,1000],
                      }
logistic_grid = GridSearchCV(model_log, logistic_param_grid, verbose=3, cv=10)


# In[53]:


# Fit the model using the grid search estimator
logistic_grid.fit(X_train_scaled, y_train)


# In[ ]:





# In[54]:


# Print scores for Logistic Regression
print(logistic_grid.best_params_)
print(logistic_grid.best_score_)



# In[63]:


from joblib import dump, load
dump(model_log, 'model.pkl')


# In[64]:


model_log = load('model.pkl')


# In[66]:


# Saving the data columns from training
model_columns = list(X_train.columns)
dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")


# In[ ]:




