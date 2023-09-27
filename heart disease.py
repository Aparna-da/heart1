
pip install seaborn 



import numpy as np
print(np.__version__)


# In[3]:


pip install --upgrade numpy


# In[4]:


pip install --upgrade numpy


# In[5]:


pip install --upgrade scikit-learn numpy


# In[6]:


import numpy as np
np.float = float


# In[7]:


pip install scikit-learn 


# In[8]:


pip show scikit-learn


# In[9]:


pip install --upgrade scikit-learn


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report
import matplotlib.pyplot as plt


# In[12]:


# df = pd.read_csv('heart_disease.csv')
# df.head()

from google.colab import auth
auth.authenticate_user()
!pip install google-cloud-storage
import pandas as pd
from google.cloud import storage
from io import StringIO

def load_data_from_gcp(bucket_name, blob_name):
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the blob content to a string
    data = blob.download_as_text()

    # Load the data into a pandas DataFrame
    df = pd.read_csv(StringIO(data))
    
    return df

# Provide your GCP bucket name and blob/file name
bucket_name = 'dataset0111'
blob_name = 'data/files/md5/f8/376ac9db4d25345aead44787474f27'
df = load_data_from_gcp(bucket_name, blob_name)

# In[13]:


df.shape


# In[14]:


df.info()


# In[15]:


df['target'].value_counts()


# In[16]:


# To view the value counts in percentage
df['target'].value_counts(normalize=True) 


# In[17]:


#plot value counts with bar graph
df['target'].value_counts().plot(kind='bar',color=['salmon','lightblue'])


# In[18]:


df.describe()


# In[19]:


df.corr()


# In[20]:


ax,fig = plt.subplots(figsize=(18,9))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')


# In[21]:


df['sex'].value_counts()


# In[22]:


pd.crosstab(df['target'],df['sex'])


# In[23]:


pd.crosstab(df['target'],df['sex']).plot(kind='bar',color=['salmon','lightblue'],figsize=(10,6))

plt.title('Heart disease frequency for sex')
plt.ylabel('Count')
plt.legend(['Female','Male'])


# In[24]:


plt.figure(figsize=(10,6))

plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="salmon") # define it as a scatter figure

# Now for negative examples, we want them on the same plot, so we call plt again
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="lightblue") # axis always come as (x, y)

plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");


# In[25]:


df['age'].plot.hist();


# In[26]:


X = df.drop('target',axis=1)

y = df['target']


# In[27]:


X.head()


# In[28]:


y


# In[29]:


np.random.seed(42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[30]:


X_train.head()


# In[31]:


X_test.head()


# In[32]:


len(X_train)


# In[33]:


len(X_test)


# In[34]:


models = {'RandomForest':RandomForestClassifier(),
         'Logistic Regression':LogisticRegression(),
         'KNN':KNeighborsClassifier(),
         'Decision Trees':DecisionTreeClassifier(),
         'Bayes':GaussianNB()}

def fit_and_score(models,X_train,X_test,y_train,y_test):
    '''
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    
    '''
    
    np.random.seed(42)

    model_scores={}
    
    for name,model in models.items():
        model.fit(X_train,y_train)
        model_scores[name] = model.score(X_test,y_test)
    return model_scores


# In[35]:


model_scores = fit_and_score(models=models,
                            X_train=X_train,
                            X_test = X_test,
                            y_train=y_train,
                            y_test = y_test)

model_scores


# In[36]:


model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();


# In[37]:


# tuning for LogisticRegression
log_reg_grid = {'C':np.logspace(-4,4,20),
               'solver':['liblinear']}

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               verbose=True,
                               n_iter=20,
                               param_distributions=log_reg_grid,
                               cv=5)


# In[38]:


rs_log_reg.fit(X_train,y_train)


# In[39]:


rs_log_reg.score(X_test,y_test)


# In[40]:


# tuning for RandomForestClassifier

rf_grid = {'n_estimators':np.arange(10,1000,50),
          'max_depth':[None,3,5,10],
          'min_samples_split':np.arange(2,20,2),
          'min_samples_leaf':np.arange(1,20,2)}

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions=rf_grid,
                          n_iter=20,cv=5,
                          verbose=True)


# In[41]:


rs_rf.fit(X_train,y_train) 


# In[42]:


rs_rf.score(X_test,y_test)


# In[43]:


model_scores


# In[44]:


rs_log_grid = RandomizedSearchCV(LogisticRegression(),
                               verbose=True,
                               param_distributions=log_reg_grid,
                               cv=5)


# In[45]:


rs_log_grid.fit(X_train,y_train)


# In[46]:


rs_log_grid.score(X_test,y_test)


# In[47]:


y_preds = rs_log_grid.predict(X_test)


# In[48]:


y_preds


# In[49]:


y_test


# In[56]:


# plot_roc_curve(rs_log_grid,X_test,y_test)

fpr, tpr, _ = roc_curve(y_test, y_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[57]:


print(confusion_matrix(y_test,y_preds))


# In[58]:


sns.heatmap(confusion_matrix(y_test,y_preds),
                            annot=True)


# In[59]:


print(classification_report(y_test,y_preds))


# In[60]:


rs_log_grid.best_params_


# In[61]:


clf = LogisticRegression(C=0.615848211066026,
                        solver='liblinear')


# In[62]:


cv_acc = cross_val_score(clf,X,y,cv=5,scoring='accuracy')
cv_acc


# In[63]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[64]:


cv_acc = cross_val_score(clf,X,y,cv=5,scoring='recall')
cv_acc


# In[65]:


cv_acc= np.mean(cv_acc)
cv_acc


# In[66]:


cv_acc = cross_val_score(clf,X,y,cv=5,scoring='f1')
cv_acc


# In[67]:


cv_acc=np.mean(cv_acc)
cv_acc


# In[68]:


import pickle
pickle.dump(rs_log_reg,open('heart.pickle','wb'))


# In[69]:


model = pickle.load(open('heart.pickle','rb'))


# In[70]:


X


# In[71]:


X.columns


# In[72]:


model.predict([[35,0,3,150,250,0,1,185,1,3.5,1,0,3]])






