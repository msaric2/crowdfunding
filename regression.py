import pandas as pd
import sys
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
import statsmodels.api as sm
import os
import scipy.stats  as stats


def descriptiveStatistics(var):
    # calculate descriptive statistics
    print data2[var].mean()
    print data2[var].std()
    print data2[var].min()
    print data2[var].max()

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# folder='c:/work/crowdfunding/download/18_5_18/'
# listing = os.listdir(folder)
# data=[]
# datafiles=[]
# counter=1
# for file in listing:
#     datafiles.append(pd.read_csv(file, header=0))
#     counter=counter+1;
#     if counter>5:
#         break
# data=pd.concat
data = pd.read_csv('selectedCEEv2.csv', header=0)  #'download/18_5_18/Kickstarter001.csv'
#data = data.dropna()
print(data.shape)
print(list(data.columns))
data=data.fillna(0)
#create dummy variables
data2 = pd.get_dummies(data, columns =['state','staff_pick','spotlight'])

#clear projects with states canceled, suspended, live
data2=data2[(data2.state_canceled==False)]
data2=data2[(data2.state_suspended==False)]
data2=data2[(data2.state_live==False)]

# drop projects with small goals
data2=data2[data2.goal>100]
data2=data2[data2.goal<1000000]

#number of successful projects
numOfSucessful=data2[(data2.state_successful==True)].__len__()
successfulProjects=data2[(data2.state_successful==True)]
#number of failed projects
numOfFailed=data2[(data2.state_failed==True)].__len__()
failedProjects=data2[(data2.state_failed==True)]


percentageFundedFailed=failedProjects['pledged']/failedProjects['goal']
weights = np.ones_like(percentageFundedFailed)/float(len(percentageFundedFailed))
bins=np.linspace(0,1,20)
plt.figure(1)
plt.hist(percentageFundedFailed, weights=weights,bins=bins)
plt.title('Pledge Levels of Failed Projects')
plt.xlabel('Percentage funded')
plt.ylabel('Percent')
percentageFundedFailed[percentageFundedFailed.iloc[:]>0.3].__len__()/float(len(percentageFundedFailed))
# plt.show()



percentageFundedSuccessful=successfulProjects['pledged']/successfulProjects['goal']
weights = np.ones_like(percentageFundedSuccessful)/float(len(percentageFundedSuccessful))
bins=np.linspace(1,2,20)
plt.figure(2)
plt.hist(percentageFundedSuccessful, weights=weights,bins=bins)
plt.xlabel('Percentage funded')
plt.ylabel('Percent')
plt.title('Pledge Levels of Successful Projects')
# plt.show()
#statistics for each country
# countries=['SI','HR','CZ','PL','BG','HU','RO','UA','SK','LV','LT','EE','RS']
countries=['SI', 'HR', 'CZ', 'PL', 'BG', 'HU', 'RO', 'AL', 'SK', 'LV', 'LT', 'EE']
successPercentage=list()
for country in countries:
    dataCountry=pd.DataFrame(columns=data2.columns)
    for i,row in data2.iterrows():
        if(row.location.find('"country":' + '"' + country)==1):   #if(row.location.find('"country":"CZ"')==1):
            dataCountry = dataCountry.append(row)
    successPercentage.append(float(dataCountry[(dataCountry.state_successful==True)].__len__())/dataCountry.__len__())

plt.figure(3)
plt.bar(countries,successPercentage)
plt.title('Percentage of Successful Projects')
plt.xlabel('country')
plt.ylabel('Percent')
plt.show()



# drop projects with small goals
# data2=data2[data2.goal>5000]
#data2=data2[data2.goal<1000000]

# add duration column

data2['duration']=(data2.deadline-data2.launched_at)/(3600.0*24)
#data2['duration']=data2['duration']/data2['duration'].max()
#data2['goal']=data2['goal']/data2['goal'].max()
data2['mean_contribution']=data2.pledged/data2.backers_count
data2['mean_contribution']=data2['mean_contribution'].fillna(0)



#calculate correlations
stats.pearsonr(data2['state_successful'],data2['backers_count'])

data2=data2[data2.goal>=5000]   #only large projects

data2['goal']=np.log10(data2['goal'])
#for i in data2['mean_contribution']:
#    if data2.backers_count.values
data3 = data2.filter(['state_successful','backers_count','goal','mean_contribution','duration' ], axis=1)  #  'pledged','duration','staff_pick_True'  state_successful 'duration'   'backers_count', 'mean_contribution',,'goal''backers_count',,'goal','duration','spotlight_True','staff_pick_True'

writer = pd.ExcelWriter('c:/work/crowdfunding/data4LargeProjects.xlsx')
data3.to_excel(writer,'sheet1')
writer.save()

#data3.at[2,'spotlight_True']    data.isnull().values.any()

X = data3.iloc[:,1:]
y = data3.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#from sklearn.metrics import confusion_matrix
#from sklearn.feature_selection import RFE
#classifier = LogisticRegression(random_state=0)
#rfe = RFE(classifier, 6)
#rfe = rfe.fit(X_train, y_train )
#print(rfe.support_)
#print(rfe.ranking_)


#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)

#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)


X=sm.add_constant(X)
logit_model=sm.Logit(y,X,missing='drop')
result=logit_model.fit()
#linear_model=sm.OLS(y,X,missing='drop')
#result=linear_model.fit()  # method='bfgs', maxiter=350
print(result.summary())
#logit_model.predict()