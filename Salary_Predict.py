# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 19:10:33 2022

@author: bsoum
"""


#import the libraries
import pandas as pd
from datetime import date
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV



#load the data into the python enviroment
data = pd.read_csv(r'C:\Users\bsoum\OneDrive\Desktop\SalaryPredictionIntern\HRDataset_v14 (1).csv')

# Data Cleaning
data= data.drop(['Employee_Name', 'EmpID','Sex', 'MaritalDesc','LastPerformanceReview_Date','ManagerID','ManagerName','LastPerformanceReview_Date','Position','Department','PerformanceScore','EmploymentStatus'],axis=1)


#fill the missing values with the current date to calculate the experience


today = date.today()
today = today.strftime('%m/%d/%Y')
data['DateofTermination'].fillna(today, inplace = True)

#calculate the experience
d2 = pd.to_datetime(data['DateofTermination'])
d1 = pd.to_datetime(data['DateofHire'])

data['YearsofExperience'] = round(((d2 - d1).dt.days / 365.25),1)

#Calculate the age using date of birth

 #for  calculating the age year in date of birth is extracted 
data['YearOfBirth']=pd.DatetimeIndex(data['DOB']).year.astype('int')
data['YearOfBirth']
def fix_time(f):
    if f>2020:
        f=f-100
    else:
        f=f
    return f
data['YearOfBirth']= data['YearOfBirth'].apply(fix_time).astype('int')
today = date.today()
data['Age']=today.year-data['YearOfBirth']
data['Age']

data = data.drop(['DateofHire', 'DateofTermination','YearOfBirth','DOB'],axis=1)

#Data Preprocessing


categ = ['State','RaceDesc','CitizenDesc','HispanicLatino','TermReason','RecruitmentSource']
le = LabelEncoder()
for i in range(len(categ)):
  data[categ[i]] = le.fit_transform(data[categ[i]])
  
# Modelling

data=data.drop(['MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID', 'FromDiversityJobFairID', 'Termd', 'State', 'Zip', 'CitizenDesc', 'HispanicLatino', 'TermReason','RecruitmentSource','EngagementSurvey','EmpSatisfaction','DaysLateLast30','Absences'],axis=1)

x= data.drop(['Salary'],axis=1)
y =data['Salary']


X_train,X_test,Y_train,Y_test=train_test_split(x,y, test_size=0.3, random_state=42)


param_grid = {'n_estimators': [100, 80, 60, 55, 51, 45],  
              'max_depth': [7, 8],
              #'reg_lambda' :[0.26, 0.25, 0.2]
             }
                
grid = GridSearchCV(GradientBoostingRegressor(), param_grid, refit = True, verbose = 3, n_jobs=-1) #
regr_trans = TransformedTargetRegressor(regressor=grid, transformer=MinMaxScaler(feature_range=(0, 1)))
# fitting the model for grid search 
grid_result=regr_trans.fit(X_train,Y_train)
best_params=grid_result.regressor_.best_params_
print(best_params)

best_model_gbr = GradientBoostingRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"])
regr_trans = TransformedTargetRegressor(regressor=best_model_gbr, transformer=MinMaxScaler(feature_range=(0, 1)))
regr_trans.fit(X_train, Y_train)
Y_pred = regr_trans.predict(X_test)
print(Y_pred)

#Saving the model to disk

pickle.dump(regr_trans,open('Salary_Predict.pkl','wb') )




