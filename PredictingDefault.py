#!/usr/bin/env python
# coding: utf-8

# In[4]:



# Taken grade as a Y variable
# Taken normalized values of numerial data as X 

#f-score: Training set given as csv:0.81
#f-score: Testing set given as csv:0.93

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from datetime import datetime, date, time, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
plt.style.use('ggplot')


# In[5]:


#Reading csv file -training and testing data to separate dataframes.
df_training=pd.read_csv('train_indessa.csv')
df_testing=pd.read_csv('test_indessa.csv')
df_training.columns
df_testing.columns


# In[6]:


#Cleaning Training data
df_training.drop(['batch_enrolled'], axis = 1)
df_training.drop(['emp_title'], axis = 1)
df_training.drop(['emp_length'], axis = 1)
df_training.drop(['desc'], axis = 1)
df_training.drop(['zip_code'], axis = 1)
df_training.drop(['addr_state'], axis = 1)
df_training.drop(['last_week_pay'], axis = 1)
df_training.drop(['loan_amnt'], axis = 1)
df_training.drop(['funded_amnt'], axis = 1)
#Cleaning Testing data

df_testing.drop(['batch_enrolled'], axis = 1)
df_testing.drop(['emp_title'], axis = 1)
df_testing.drop(['emp_length'], axis = 1)
df_testing.drop(['desc'], axis = 1)
df_testing.drop(['zip_code'], axis = 1)
df_testing.drop(['addr_state'], axis = 1)
df_testing.drop(['last_week_pay'], axis = 1)
df_testing.drop(['loan_amnt'], axis = 1)
df_testing.drop(['funded_amnt'], axis = 1)




#Taking relevant columns- Numeric columns
df_train= df_training[['member_id','grade','funded_amnt_inv','term','int_rate','annual_inc','dti',
        'delinq_2yrs','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal',
        'revol_util','total_acc','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee',
        'collections_12_mths_ex_med','mths_since_last_major_derog','acc_now_delinq','tot_coll_amt',
        'tot_cur_bal','total_rev_hi_lim']]

df_test= df_testing[['member_id','grade','funded_amnt_inv','term','int_rate','annual_inc','dti',
        'delinq_2yrs','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal',
        'revol_util','total_acc','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee',
        'collections_12_mths_ex_med','mths_since_last_major_derog','acc_now_delinq','tot_coll_amt',
        'tot_cur_bal','total_rev_hi_lim']] 

#Replacing NaN values with 0
df_train=df_train.fillna(0)
df_test=df_test.fillna(0)
df_test[:10]


# In[6]:


df_train[:10]


# In[7]:


df_test.columns


# In[7]:


import re
#Using regex to replace text like months and convert the column , retaining float value.
df_train_=df_train.replace('months',"",regex=True)
df_train_.dtypes
df_train_['term'] = df_train_['term'].astype(float, errors = 'raise')
df_train_.dtypes

df_test_=df_test.replace('months',"",regex=True)

df_test_.dtypes

df_test_['term'] = df_test_['term'].astype(float, errors = 'raise')
df_test_.dtypes


df_train_['member_id']=df_train_['member_id'].astype(str, errors = 'raise')
df_test_['member_id']=df_test_['member_id'].astype(str, errors = 'raise')

#Fill NaN with 0, to handle any empty value after regex process
df_train_=df_train_.fillna(0)
df_test_=df_test_.fillna(0)
df_train_['funded_amnt_inv'] = df_train_['funded_amnt_inv'].div(df_train_['funded_amnt_inv'].sum()).multiply(100)
df_train_['annual_inc'] = df_train_['annual_inc'].div(df_train_['annual_inc'].sum()).multiply(100)
df_train_['total_rec_int'] = df_train_['total_rec_int'].div(df_train_['total_rec_int'].sum()).multiply(100)
df_train_['tot_cur_bal'] = df_train_['tot_cur_bal'].div(df_train_['tot_cur_bal'].sum()).multiply(100)
df_train_['total_rev_hi_lim'] = df_train_['total_rev_hi_lim'].div(df_train_['total_rev_hi_lim'].sum()).multiply(100)
df_train_['tot_coll_amt'] = df_train_['tot_coll_amt'].div(df_train_['tot_coll_amt'].sum()).multiply(100)
df_train_['dti'] = df_train_['dti'].div(df_train_['dti'].sum()).multiply(100)
df_train_['revol_bal'] = df_train_['revol_bal'].div(df_train_['revol_bal'].sum()).multiply(100)
df_train_['revol_util'] = df_train_['revol_util'].div(df_train_['revol_util'].sum()).multiply(100)
df_train_['total_acc'] = df_train_['total_acc'].div(df_train_['total_acc'].sum()).multiply(100)
df_train_['total_rec_late_fee'] = df_train_['total_rec_late_fee'].div(df_train_['total_rec_late_fee'].sum()).multiply(100)
df_train_['recoveries'] = df_train_['recoveries'].div(df_train_['recoveries'].sum()).multiply(100)
df_train_['collection_recovery_fee'] = df_train_['collection_recovery_fee'].div(df_train_['collection_recovery_fee'].sum()).multiply(100)
df_train_['acc_now_delinq'] = df_train_['acc_now_delinq'].div(df_train_['acc_now_delinq'].sum()).multiply(100)
df_train_['tot_cur_bal'] = df_train_['tot_cur_bal'].div(df_train_['tot_cur_bal'].sum()).multiply(100)
df_train_['int_rate'] = df_train_['int_rate'].div(df_train_['int_rate'].sum()).multiply(100)
df_train_['open_acc'] = df_train_['open_acc'].div(df_train_['open_acc'].sum()).multiply(100)
df_train_['pub_rec'] = df_train_['pub_rec'].div(df_train_['pub_rec'].sum()).multiply(100)
df_train_['delinq_2yrs'] = df_train_['delinq_2yrs'].div(df_train_['delinq_2yrs'].sum()).multiply(100)
df_train_['inq_last_6mths'] = df_train_['inq_last_6mths'].div(df_train_['inq_last_6mths'].sum()).multiply(100)
df_train_['acc_now_delinq'] = df_train_['acc_now_delinq'].div(df_train_['acc_now_delinq'].sum()).multiply(100)


df_train_['term'] =df_train_['term'].div(1200).round(2)
df_train_['mths_since_last_delinq'] =df_train_['mths_since_last_delinq'].div(1200).round(5)
df_train_['mths_since_last_major_derog'] =df_train_['mths_since_last_major_derog'].div(1200).round(5)
df_train_['collections_12_mths_ex_med'] =df_train_['collections_12_mths_ex_med'].div(1200).round(5)
df_train_['mths_since_last_delinq'] =df_train_['mths_since_last_delinq'].div(1200).round(5)
df_train_['mths_since_last_record'] =df_train_['mths_since_last_record'].div(1200).round(5)

df_test_['funded_amnt_inv'] = df_test_['funded_amnt_inv'].div(df_test_['funded_amnt_inv'].sum()).multiply(100)
df_test_['annual_inc'] = df_test_['annual_inc'].div(df_test_['annual_inc'].sum()).multiply(100)
df_test_['total_rec_int'] = df_test_['total_rec_int'].div(df_test_['total_rec_int'].sum()).multiply(100)
df_test_['tot_cur_bal'] = df_test_['tot_cur_bal'].div(df_test_['tot_cur_bal'].sum()).multiply(100)
df_test_['total_rev_hi_lim'] = df_test_['total_rev_hi_lim'].div(df_test_['total_rev_hi_lim'].sum()).multiply(100)
df_test_['tot_coll_amt'] = df_test_['tot_coll_amt'].div(df_test_['tot_coll_amt'].sum()).multiply(100)
df_test_['dti'] = df_test_['dti'].div(df_test_['dti'].sum()).multiply(100)
df_test_['revol_bal'] = df_test_['revol_bal'].div(df_test_['revol_bal'].sum()).multiply(100)
df_test_['revol_util'] = df_test_['revol_util'].div(df_test_['revol_util'].sum()).multiply(100)
df_test_['total_acc'] = df_test_['total_acc'].div(df_test_['total_acc'].sum()).multiply(100)
df_test_['total_rec_late_fee'] = df_test_['total_rec_late_fee'].div(df_test_['total_rec_late_fee'].sum()).multiply(100)
df_test_['recoveries'] = df_test_['recoveries'].div(df_test_['recoveries'].sum()).multiply(100)
df_test_['collection_recovery_fee'] = df_test_['collection_recovery_fee'].div(df_test_['collection_recovery_fee'].sum()).multiply(100)
df_test_['acc_now_delinq'] = df_train_['acc_now_delinq'].div(df_test_['acc_now_delinq'].sum()).multiply(100)
df_test_['tot_cur_bal'] = df_test_['tot_cur_bal'].div(df_test_['tot_cur_bal'].sum()).multiply(100)
df_test_['int_rate'] = df_test_['int_rate'].div(df_test_['int_rate'].sum()).multiply(100)
df_test_['open_acc'] = df_test_['open_acc'].div(df_test_['open_acc'].sum()).multiply(100)
df_test_['pub_rec'] = df_test_['pub_rec'].div(df_test_['pub_rec'].sum()).multiply(100)
df_test_['delinq_2yrs'] = df_test_['delinq_2yrs'].div(df_test_['delinq_2yrs'].sum()).multiply(100)
df_test_['inq_last_6mths'] = df_test_['inq_last_6mths'].div(df_test_['inq_last_6mths'].sum()).multiply(100)
df_test_['acc_now_delinq'] = df_test_['acc_now_delinq'].div(df_test_['acc_now_delinq'].sum()).multiply(100)


df_test_['term'] =df_test_['term'].div(1200).round(2)
df_test_['mths_since_last_delinq'] =df_test_['mths_since_last_delinq'].div(1200).round(5)
df_test_['mths_since_last_major_derog'] =df_test_['mths_since_last_major_derog'].div(1200).round(5)
df_test_['collections_12_mths_ex_med'] =df_test_['collections_12_mths_ex_med'].div(1200).round(5)
df_test_['mths_since_last_delinq'] =df_test_['mths_since_last_delinq'].div(1200).round(5)
df_test_['mths_since_last_record'] =df_test_['mths_since_last_record'].div(1200).round(5)
df_train_


# In[35]:


# With Training set
from sklearn.model_selection import train_test_split

X = df_train_.loc[:, "funded_amnt_inv":,]
y = df_train_.loc[:,  'grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


X_test
y
X


# In[40]:



from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="auto",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight='balanced',
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)




# In[41]:


y_pred


# In[42]:


print(classification_report(y_test,y_pred))


# In[43]:


from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y_test, y_pred, average='weighted')


# In[8]:


#with Test Set
from sklearn.model_selection import train_test_split

X = df_test_.loc[:, "funded_amnt_inv":,]
y = df_test_.loc[:,  'grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


X_test
y


# In[45]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="auto",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight='balanced',
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[47]:


y_pred


# In[48]:


print(classification_report(y_test,y_pred))


# In[53]:


from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y_test, y_pred, average='weighted')


# In[ ]:





# In[ ]:




