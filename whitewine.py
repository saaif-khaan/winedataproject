#!/usr/bin/env python
# coding: utf-8

# In[441]:


import os
print(os.path)



# In[369]:


import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn as skn

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[370]:


df = pd.read_csv('F:\Data Science\python\wine\whitewine.csv')


# In[371]:


df.head(10)


# In[372]:


df.shape


# In[373]:


df.info()


# In[374]:


dfex = pd.read_csv('F:\Data Science\python\wine\whitewine4rows.csv')


# In[375]:


dfex.head()


# In[376]:


columns_to_modify = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates', 'alcohol']

# last two digit of my id is 18 for better calculation result we use desimal point
id_number = 0.18
dfex[columns_to_modify] = dfex[columns_to_modify].add(id_number)


# In[377]:


dfex.head()


# In[378]:


df = pd.concat([df, pd.DataFrame(dfex)], ignore_index=True)


# In[379]:


df.head()


# In[380]:


df.shape


# In[381]:


df.describe().T


# In[382]:


df.isnull().sum()


# In[383]:


# Let’s impute the missing values by means as the data present in the different columns are continuous values.
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
 
df.isnull().sum().sum()


# In[384]:


df.isnull().sum()


# In[385]:


sns.set(style="whitegrid")
df.hist(bins=20, figsize=(10, 10), edgecolor='black')
plt.show()


# In[386]:


sns.set(style="whitegrid")
fig, ax1 = plt.subplots(4,3, figsize=(15,30))
k = 0
columns = list(df.columns)
for i in range(4):
    for j in range(3):
            sns.boxplot(x=df['quality'], y=df[columns[k]], ax = ax1[i][j], palette='pastel')
            k += 1
plt.show()


# In[387]:


sns.set(style='whitegrid')
plt.figure(figsize=(20, 18))
sns.boxplot(df, palette='Set3')
plt.xticks(rotation=45)
plt.show()


# In[388]:


# Remove outliers. so that we can get a better result

lower_limit = df['free sulfur dioxide'].mean() - 3 * df['free sulfur dioxide'].std()
upper_limit = df['free sulfur dioxide'].mean() + 3 * df['free sulfur dioxide'].std()

print(lower_limit, upper_limit)


# In[389]:


df_sulfur_without_outliers = df[(df['free sulfur dioxide'] > lower_limit) & (df['free sulfur dioxide'] < upper_limit)]


df_sulfur_without_outliers.head(5)


# In[390]:


print(df.shape[0], df_sulfur_without_outliers.shape[0])

# difference between the two dataframes
print(df.shape[0] - df_sulfur_without_outliers.shape[0])


# In[391]:


# remove outliers from total sulfur dioxide
total_sulfur_lower_limit = df['total sulfur dioxide'].mean() - 3 * df['total sulfur dioxide'].std()
total_sulfur_upper_limit = df['total sulfur dioxide'].mean() + 3 * df['total sulfur dioxide'].std()

print(total_sulfur_lower_limit, total_sulfur_upper_limit)


# In[392]:


total_sulfur_df_without_outliers = df_sulfur_without_outliers[(df_sulfur_without_outliers['total sulfur dioxide']> total_sulfur_lower_limit) & (df_sulfur_without_outliers['total sulfur dioxide'] < total_sulfur_upper_limit)]

total_sulfur_df_without_outliers.head(5)


# In[393]:


df_sulfur_without_outliers.shape[0] - total_sulfur_df_without_outliers.shape[0]


# In[394]:


# remove outliers from residual sugar
residual_sugar_lower_limit = df['residual sugar'].mean() - 3 * df['residual sugar'].std()
residual_sugar_upper_limit = df['residual sugar'].mean() + 3 * df['residual sugar'].std()

print(residual_sugar_lower_limit, residual_sugar_upper_limit)


# In[395]:


residual_sugar_df_without_outliers = total_sulfur_df_without_outliers[(total_sulfur_df_without_outliers['residual sugar'] > residual_sugar_lower_limit) & (total_sulfur_df_without_outliers['residual sugar'] < residual_sugar_upper_limit)]

residual_sugar_df_without_outliers.head(5)


# In[396]:


total_sulfur_df_without_outliers.shape[0] - residual_sugar_df_without_outliers.shape[0]


# In[397]:


residual_sugar_df_without_outliers.isnull().sum()


# In[398]:


# this is work main df without outliers 
df = residual_sugar_df_without_outliers


# In[399]:


sns.pairplot(df)


# In[400]:


color = sns.color_palette("pastel")

fig, ax1 = plt.subplots(4,3, figsize=(20,15))
k = 0
columns = list(df.columns)
for i in range(4):
    for j in range(3):
            sns.distplot(df[columns[k]], ax = ax1[i][j], color='blue')
            k += 1
plt.show()


# In[401]:


def log_transform(col):
    return np.log(col[0])

df['residual sugar'] = df[['residual sugar']].apply(log_transform, axis=1)
df['chlorides'] = df[['chlorides']].apply(log_transform, axis=1)
df['free sulfur dioxide'] = df[['free sulfur dioxide']].apply(log_transform, axis=1)
df['total sulfur dioxide'] = df[['total sulfur dioxide']].apply(log_transform, axis=1)
df['sulphates'] = df[['sulphates']].apply(log_transform, axis=1)


# In[402]:


color = sns.color_palette("pastel")

fig, ax1 = plt.subplots(4,3, figsize=(20,15))
k = 0
columns = list(df.columns)
for i in range(4):
    for j in range(3):
            sns.distplot(df[columns[k]], ax = ax1[i][j], color='green')
            k += 1
plt.show()


# In[403]:


#Now let’s draw the count plot to visualise the number data for each quality of wine.
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[404]:


plt.figure(figsize=(12, 12))
sns.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[405]:


quality_ = df['quality'].value_counts()
plt.figure(figsize=(8,6))
plt.xlabel('Quality')
sns.barplot(x=quality_.index, y=quality_.values, alpha=0.8, palette = sns.color_palette("pastel"))
plt.show()


# In[406]:


df.corr()['quality'].sort_values(ascending=False)


# In[407]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot =True, cmap='coolwarm')


# In[408]:


corr = df.corr()[['quality']]
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[409]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr() > 0.7, annot=True, cbar=False, cmap='coolwarm' )
plt.show()


# In[410]:


# relationship between wine and other characteristics
df.corrwith(df['quality']).plot.bar(figsize=(10,8), title='Correlation with quality', rot=45, grid=True, fontsize=14)


# In[411]:


df.quality.value_counts()


# In[412]:


#  0 - 5 = bad wine 
# 6 - 10 = good 

# first convert quality to int
df['quality'] = df['quality'].astype(int)

# list of quality in string
quality_list = { 0:'bad', 1:'bad', 2:'bad', 3: 'bad', 4: 'bad', 5: 'bad', 6: 'good', 7: 'good', 8: 'good', 9: 'good', 10: 'good'}

# map quality to string
df['quality'] = df['quality'].map(quality_list)

print(df['quality'].value_counts())


# In[413]:


# df['quality'].value_counts()
df.head(5)


# In[414]:


sns.countplot(data = df, x = 'quality')
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()


# In[415]:


# alcohol level for distribution of class of alcohol data
fig, ax = plt.subplots(1,2, figsize=(12,5))
sns.histplot(data=df, x='alcohol', ax=ax[0], color='teal')
sns.boxplot(data=df, x='quality', y='alcohol', ax=ax[1], color='teal')
ax[0].set_title('Distribution of Alcohol', fontsize=14)
ax[1].set_title('Alcohol vs Qaulity', fontsize=14)
plt.show()


# In[416]:


# percentage of concentrated alcohol in wine
# df.groupby('alcohol')['quality'].mean().plot.bar(figsize=(10,8), color='teal')
df.groupby('alcohol')['quality'].value_counts().sort_values(ascending=False)[0:5]
# shows that 9.5% alcohol is the most common in the dataset and 9.4% is the second most common


# In[417]:


df.isnull().sum()


# In[418]:


df['quality'] = df['quality'].map({'bad': 0, 'good': 1})


# In[419]:


print(df.dtypes)


# In[420]:


import statsmodels.api as sm
sm.Logit(df['quality'], df.drop('quality', axis=1)).fit().summary()


# In[421]:


# # Dividing dependent and independent variables

X = df.drop("quality", axis=1)
y = df["quality"]


# In[422]:


# train-test-split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1118)


# In[423]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[424]:


from sklearn.linear_model import LogisticRegression  

classifier= LogisticRegression(random_state=1118)  
lr_model=classifier.fit(X_train, y_train)
y_pred=lr_model.predict(X_test)


# In[425]:


# Accuracy = TP+TN/TP+FP+FN+TN
# Precision = TP/TP+FP
# Recall = TP/TP+FN
#F1 Score = 2*(Recall * Precision) / (Recall + Precision)


from sklearn.metrics import classification_report, confusion_matrix 
print(classification_report(y_test, y_pred))


# In[426]:


x_axis_labels = ['Bad','Good']
y_axis_labels = ['Bad','Good']

confusion_m=confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_m, annot=True,cmap='Reds', xticklabels=x_axis_labels, yticklabels=y_axis_labels)


plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')


# In[427]:


print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('Precision: ', metrics.precision_score(y_test, y_pred))
print('Recall (Sensitivity): ', metrics.recall_score(y_test, y_pred))
print('f1-score: ', metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print('Specificity: ', specificity)

# need sensitivity and specificity


# In[428]:


#decision tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
dt_model = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt_model.predict(X_test)


# In[429]:


print(classification_report(y_test, y_pred))


# In[430]:


x_axis_labels = ['Bad','Good']
y_axis_labels = ['Bad','Good']

confusion_m=confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_m, annot=True,cmap='Reds', xticklabels=x_axis_labels, yticklabels=y_axis_labels)


plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')


# In[431]:


print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('Precision: ', metrics.precision_score(y_test, y_pred))
print('Recall (Sensitivity): ', metrics.recall_score(y_test, y_pred))
print('f1-score: ', metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print('Specificity: ', specificity)


# In[432]:


#Random forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100,random_state=0)
rf_model=clf.fit(X_train, y_train)
y_pred=rf_model.predict(X_test)


# In[433]:


print(classification_report(y_test, y_pred))


# In[434]:


x_axis_labels = ['Bad','Good']
y_axis_labels = ['Bad','Good']

confusion_m=confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_m, annot=True,cmap='Reds', xticklabels=x_axis_labels, yticklabels=y_axis_labels)


plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')


# In[435]:


print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('Precision: ', metrics.precision_score(y_test, y_pred))
print('Recall (Sensitivity): ', metrics.recall_score(y_test, y_pred))
print('f1-score: ', metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print('Specificity: ', specificity)


# In[436]:


from sklearn.svm import SVC


# Create an SVM classifier
svm_classifier = SVC(random_state=0)

# Fit the SVM model to the training data
svm_model = svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm_model.predict(X_test)


# In[437]:


print(classification_report(y_test, y_pred))


# In[438]:


x_axis_labels = ['Bad','Good']
y_axis_labels = ['Bad','Good']

confusion_m=confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_m, annot=True,cmap='Reds', xticklabels=x_axis_labels, yticklabels=y_axis_labels)


plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')


# In[439]:


print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('Precision: ', metrics.precision_score(y_test, y_pred))
print('Recall (Sensitivity): ', metrics.recall_score(y_test, y_pred))
print('f1-score: ', metrics.f1_score(y_test, y_pred))
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print('Specificity: ', specificity)


# In[ ]:





# In[ ]:




