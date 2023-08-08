#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pycaret autoviz')


# In[2]:


from pycaret.utils import version
version()


# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from pycaret.classification import *
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # Setup

# ## Feature Selection
# 
# Features will be selected using a permutation of importance techniques including Random Forest, Adaboost, and Linear correlation with the target variable.
# 
# This feature selection process comes out of the box ready with the setup of PyCaret.

# In[4]:


df = pd.read_csv(r"C:\Users\Hp\Downloads\healthcare-dataset-stroke-data (2).csv")
df.head()


# The column 'id' is irrelevant for the purpose of training our model.

# In[5]:


df = df.drop('id', axis=1)
df.head()


# In[6]:


df.isnull().sum()


# There are 201 missing entries under the BMI column.
# We will use an imputer to resolve this.

# In[7]:


df.describe()


# In[8]:


sns.countplot(y = df['stroke'])


# In[9]:


df['stroke'].value_counts()


# The target variable is highly imbalanced.
# Synthetic Minority Oversampling Technique (SMOTE) will be considered when developing the model. 

# In[10]:


sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, linewidth=0.1)


# There is some collinearity amongst the numerical variables in this dataset. There is no variable with great significant direct correlation with the target variable.

# # Exploratory Data Analysis

# In[11]:


from pycaret.classification import *

session = setup(df, target='stroke', 
                feature_selection=True,
                imputation_type="iterative",
                numeric_iterative_imputer='rf',
                iterative_imputation_iters=5,
                normalize=True, 
                remove_multicollinearity=True, 
                fix_imbalance=True,
                profile=True,
                silent=True)


# In[12]:


eda(display_format='svg')


# # Model Development

# ## Comparing Models

# PyCaret provides a function that runs the dataset into many different algorithms to have a high level to give us a high level understanding of which algorithm would be the most effective for this dataset.

# In[13]:


compare_models(sort='F1')


# ## Model Tuning

# In[14]:


lr = create_model('lr')


# In[15]:


tuned_lr = tune_model(lr, choose_better=True, optimize="F1")


# In[16]:


ada = create_model('ada')


# In[17]:


tuned_ada = tune_model(ada, choose_better=True, optimize="F1")


# In[18]:


lda = create_model('lda')


# In[19]:


tuned_lda = tune_model(lda, choose_better=True, optimize="F1")


# ## Model Analysis

# ### Logistic Regression

# In[20]:


plot_model(tuned_lr, 'auc')


# In[21]:


plot_model(tuned_lr, 'feature')


# In[22]:


plot_model(tuned_lr, 'confusion_matrix')


# ### Ada Boost Classifier

# In[23]:


plot_model(tuned_ada, 'auc')


# In[24]:


plot_model(tuned_ada, 'feature')


# In[25]:


plot_model(tuned_ada, 'confusion_matrix')


# ### Linear Discriminant Analysis

# In[26]:


plot_model(tuned_lda, 'auc')


# In[27]:


plot_model(tuned_lda, 'feature')


# In[28]:


plot_model(tuned_lda, 'confusion_matrix')


# ## Stacking Models

# In[29]:


stacked_model = stack_models([tuned_lr, tuned_lda, tuned_ada])


# In[30]:


stacked_model = tune_model(stacked_model, choose_better=True)


# In[31]:


plot_model(stacked_model, 'confusion_matrix')


# ## Test Model

# We will now test the stacked model with test data we saved from our setup.

# In[32]:


predict_model(stacked_model)


# The performance of our model really good.
# It has high accuracy with a high Area Under Curve.
# The recall rate is not as high as we want it to be, but it is hard to keep it high as the accuracy of our model goes higher.
# 
# We will now finalize the model by training the model with the test data.

# ## Finalizing Model

# In[33]:


stacked_final = finalize_model(stacked_model)


# In[34]:


stacked_final


# Our model is finalized and ready to be saved into a pickle file.

# ## Saving the Model

# In[35]:


save_model(stacked_final, './stacked_final_stroke_predict')

