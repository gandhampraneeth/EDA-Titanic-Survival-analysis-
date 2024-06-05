#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


titanic_data=pd.read_csv('/Users/gandhampraneeth/Desktop/Titanic Analysis/titanic.csv')
titanic_data


# In[22]:


titanic_data.head()


# In[23]:


titanic_data.tail()


# In[25]:


titanic_data.describe


# In[26]:


titanic_data.info


# In[27]:


titanic_data.isnull().sum()


# In[28]:


plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['Age'].dropna(), kde=False, bins=30, color='blue')
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[29]:


# Visualize the survival rate by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title('Survival Count by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Sex', loc='upper right')
plt.show()


# In[30]:


# Visualize the survival rate by passenger class
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Pclass', loc='upper right')
plt.show()



# In[31]:


# Visualize the survival rate by embarkation port
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Embarked', data=titanic_data)
plt.title('Survival Count by Embarkation Port')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Embarked', loc='upper right')
plt.show()


# In[32]:


# Visualize the survival rate by number of siblings/spouses aboard
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='SibSp', data=titanic_data)
plt.title('Survival Count by Siblings/Spouses Aboard')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='SibSp', loc='upper right')
plt.show()



# In[34]:


# Visualize the survival rate by number of parents/children aboard
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Parch', data=titanic_data)
plt.title('Survival Count by Parents/Children Aboard')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.legend(title='Parch', loc='upper right')
plt.show()


# In[ ]:




