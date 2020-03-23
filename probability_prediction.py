# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:55:10 2020

@author: LENOVO
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline


df_train = pd.read_excel('Train_dataset.xlsx')
df_test = pd.read_excel('Test_dataset.xlsx')

Variable_Description = pd.read_excel('Variable_Description.xlsx')

df_test.info()
df_train.info()



Variable_list = Variable_Description["Variables"].tolist()
Variable_list

df_train1 = df_train
df_train1 =df_train1.drop(['people_ID','Designation','Name','Occupation', 'Pulmonary score','Region'], axis=1)


df_test1 = df_test
df_test1 =df_test1.drop(['people_ID','Designation','Name','Occupation', 'Pulmonary score','Region'], axis=1)
df_test1.isnull().sum()

''' list1 = [Children,Mode_transport,comorbidity,cardiological pressure,HBB,HDL cholesterol] - apply fillna'''
list1 = ['Children','Mode_transport','comorbidity','cardiological pressure','HBB','HDL cholesterol']
for i in list1:
    df_train1[i].fillna( method = 'ffill', inplace = True)

null_values=df_train1.isnull().sum()

#'''Pulmonary score - remove '<' '''
#df_train1['Pulmonary score'].describe()
#df_train1['Pulmonary score']=df_train1['Pulmonary score'].str.replace('<','')
#df_test1['Pulmonary score'].describe()
#df_test1['Pulmonary score']=df_test1['Pulmonary score'].str.replace('<','')
'''    
'''case of dropping the nan values using dropna'''    
df_train2 = df_train1
df_train2.dropna(inplace = True)
df_train2.isnull().sum()
    

'''case of dropping the row, the 10% missing term'''    
 df_train1.drop([ ], inplace=True,axis=1)   
 '''
 
 '''replace the missing data by most frequent value'''
df_train['FT/month'].describe() 
 df_train1['FT/month'].fillna(1, inplace = True) 
df_train1.isnull().sum()




''' list2 = ['Diuresis','Platelets','d-dimer','Heart rate','Insurance'] - missing values about 10% of data'''
'''replacing the missing data by mean'''
list2 = ['Diuresis','Platelets','d-dimer','Heart rate','Insurance']
for i in list2:
    df_train1[i].fillna(df_train1[i].mean(), inplace=True)
df_train1.isnull().sum()

'''Categorical Data Analysis'''


''' binary_categorical = ['Gender','Married']'''
''' categorical = ['Region', 'Mode_transport', 'comorbidity','cardiological pressure']'''
#df_train1['Pulmonary score'].describe()
#df_test1['Pulmonary score'].describe()

'''categorical = ['Gender','Married','Region', 'Mode_transport', 'comorbidity','cardiological pressure','Pulmonary score']'''
df_obj = df_train1.select_dtypes(include=['object']).copy()
list3 = list(df_obj[:])
df_obj = pd.get_dummies(df_obj[:], drop_first=True)
df_train1.drop(list3, inplace=True,axis=1) 
df_obj
df_train1

df_obj_test = df_test1.select_dtypes(include=['object']).copy()
df_obj_test = pd.get_dummies(df_obj_test[:], drop_first=True)
df_test1.drop(list3, inplace=True,axis=1) 

#final dataframe
df = pd.merge(df_obj, df_train1,right_index=True, left_index=True)
df_test_fin = pd.merge(df_obj_test, df_test1,right_index=True, left_index=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




df2 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df2
df1 = df2.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)
print(f"r2: {round(r2, 4)}")


from sklearn.preprocessing import StandardScaler
sc_X_1 = StandardScaler()
df_test_fin = sc_X_1.fit_transform(df_test_fin)

y_pred_prob = regressor.predict(df_test_fin)









