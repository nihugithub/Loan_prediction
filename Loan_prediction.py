#Nihal_Load_prediction_model_python

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("N:/Python projects/training_data.csv")


# In[5]:


df.head(10)


# In[6]:


df.describe()


# In[8]:


df["Property_Area"].value_counts()


# In[12]:


df["ApplicantIncome"].hist(bins=60)


# In[13]:


df.boxplot(column = "ApplicantIncome")


# In[14]:


df.boxplot(column = "ApplicantIncome", by = "Education")


# In[15]:


df["LoanAmount"].hist(bins=60)


# In[16]:


df.boxplot(column = "LoanAmount")


# In[34]:


temp1 = df["Credit_History"].value_counts(ascending=True)
temp2 = df.pivot_table(values = 'Loan_Status',index = ["Credit_History"],aggfunc= lambda x: x.map({"Y":1,"N":0}).mean())
print("frequency for credit history")
print(temp1)
print("pivot table")
print(temp2)


# In[41]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,4))
ax1= fig.add_subplot(121)
temp1.plot(kind = "bar")

ax2 = fig.add_subplot(122)
temp2.plot(kind="bar")


# In[55]:


temp3 = pd.crosstab(df["Credit_History"],df["Loan_Status"])
temp3.plot(kind='bar',stacked=True)


# In[48]:


#Finding null values in each column


# In[56]:


df.apply(lambda x:sum(x.isnull()),axis=0)


# In[50]:


#filling LoanAmount using mean values


# In[57]:


df["LoanAmount"].fillna(df['LoanAmount'].mean(), inplace =True)


# In[67]:


df.boxplot(column='LoanAmount', by = 'Education'),df.boxplot(column='LoanAmount', by = 'Self_Employed')


# In[68]:


#checking missing values in employment


# In[69]:


df["Self_Employed"].value_counts()


# In[70]:


#replacing self employed missing values with NO.


# In[71]:


df["Self_Employed"].fillna("No",inplace = True)


# In[72]:


df["Self_Employed"].value_counts()


# In[73]:


# Create a pivot table


# In[83]:


table = df.pivot_table(values = 'LoanAmount', index = 'Self_Employed', columns = 'Education', aggfunc = np.median)


# In[84]:


table


# In[85]:


df["LoanAmount"].value_counts()


# In[86]:


#applying log to LoanAmount to nullify the effect of outliers


# In[88]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins = 20)


# In[89]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 


# In[90]:


# Replacing missing values in Gender, Married, Dependents, Loan_Amount_Term, Credit_History


# In[95]:


df["Gender"].value_counts()


# In[96]:


df["Gender"].fillna("Male",inplace = True)


# In[97]:


df["Married"].value_counts()


# In[98]:


df["Married"].fillna("Yes",inplace = True)


# In[99]:


df["Dependents"].value_counts()


# In[100]:


df["Dependents"].fillna(0,inplace = True)


# In[101]:


df["Loan_Amount_Term"].value_counts()


# In[102]:


df["Loan_Amount_Term"].fillna(360.0,inplace = True)


# In[103]:


df["Credit_History"].value_counts()


# In[104]:


df["Credit_History"].fillna(1.0,inplace= True)


# In[105]:


df.head(10)


# In[106]:


#converting all categorical values into numerical using python function


# In[111]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes


# In[112]:


df.head(10)


# In[113]:


#Importing required libraries and modules to perform analysis


# In[124]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
#from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[125]:


#creating a generic function for model input and output


# In[135]:


def classification_model(model,data,predictors,outcome):
    model.fit(data[predictors],data[outcome]) #model fit
    predictions = model.predict(data[predictors]) #predict on training set
    accuracy = metrics.accuracy_score(predictions,data[outcome]) #measuring accuracy
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    #k-fold crossvaludation with k=5
    kf = KFold(data.shape[0],n_folds = 5)
    error = []
    for train,test in kf:
        #filtering training data
        train_predictors = (data[predictors].iloc[train,:])
        
        #target we using to train algorithm
        train_target = data[outcome].iloc[train]
        
        #error record from each cv run
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[train]))
    
    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome]) 
    


# In[136]:


#performing logistic regression with credit history


# In[134]:


outcome_var = "Loan_Status"
model = LogisticRegression()
predictor_var = ["Credit_History"]
classification_model(model,df,predictor_var,outcome_var)


# In[ ]:




