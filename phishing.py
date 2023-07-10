
# Detection of phishing websites

## 1. Objectives
# A phishing website is a common social engineering method that mimics trustful uniform resource locators (URLs) and webpages. 
# The objective of this notebook is to train machine learning models and deep neural nets on the dataset created to predict phishing websites. 
# Both phishing and benign URLs of websites are gathered to form a dataset and from them required URL and website content-based features are extracted. 
# The performance level of each model is measures and compared.


## 2. Data collections
# For this project, we need a bunch of urls of type legitimate (0) and phishing (1).

# The collection of phishing urls is rather easy because of the opensource service called PhishTank. 
# This service provide a set of phishing URLs in multiple formats like csv, json etc. that gets updated hourly. 
# To download the data: https://www.phishtank.com/developer_info.php

# For the legitimate URLs, I found a source that has a collection of benign, spam, phishing, malware & defacement URLs. The source of the dataset is University of New Brunswick, https://www.unb.ca/cic/datasets/url-2016.html. 
# The number of legitimate URLs in this collection are 35,300. 
# The URL collection is downloaded & from that, 'Benign_list_big_final.csv' is the file of our interest. 
# This file is then uploaded to the Colab for the feature extraction.


## 3. Collection of URL dataset
### 3.1. Phishing urls dataset



import pandas as pd
phish_data = pd.read_csv("data/full_phishing.csv")
phish_data.head()
phish_data.tail()
phish_data.shape


#Collecting 1000 Phishing URLs randomly
phish_url = phish_data.sample(n = 1000, random_state = 12).copy()
phish_url = phish_url.reset_index(drop=True)
phish_url.head()
phish_url.shape


### 3.2. Legitimate urls dataset
legit_data = pd.read_csv("data/full_legimate.csv")
legit_data.head()
legit_data.shape


#Collecting 1000 Legitimate URLs randomly
legi_url = legit_data.sample(n = 1000, random_state = 12).copy()
legi_url = legi_url.reset_index(drop=True)
legi_url.head()
legi_url.shape


## 4. Feature Extraction

### 4.1. URL-based features
# - domain of url
# - IP address in URL
# - length of url
# - depth of url
# - "http/https" in Domain
# - Using URL Shortening Services “TinyURL”
# - Count of prefix or sufix "-" in Domain
# - Count of prefix or sufix "_" in Domain
# - Sub-domain length
# - "client" in string
# - "admin" in string
# - "login" in string
# - "server" in string

# importing required packages for this section
from urllib.parse import urlparse,urlencode
import ipaddress
import re


#### 4.1.1. Domain of url
# We are just extracting the domain present in the URL. This feature doesn't have much significance in the training. May even be dropped while training the model.

def getDomain(url):
    domain = urlparse(url).netloc
    if re.match(r"^www.",domain):
        domain = domain.replace("www.","")
    return domain

#### 4.1.2. Length of URL
# Phishing URLs may have excessively long domains or domain strings that appear unusual or unrelated to the legitimate domain.

def lengthURL(url):
    return len(url)

#### 4.1.3. Depth of URL
# Computes the depth of the URL. This feature calculates the number of sub pages in the given url based on the '/'.

# The value of feature is a numerical based on the URL.

def depthURL(url):
    s = urlparse(url).path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j])!=0:
            depth = depth+1
    return depth

#### 4.1.4. "http/https" in Domain name
# Checks for the presence of "http/https" in the domain part of the URL. The phishers may add the “HTTPS” token to the domain part of a URL in order to trick users.

# If the URL has "http/https" in the domain part, the value assigned to this feature is 1 (phishing) or else 0 (legitimate).

def httpDomain(url):
    if "https" in url:
        return 1
    else:
        return 0

#### 4.1.5. URL shortening
# URL shortening is a method on the “World Wide Web” in which a URL may be made considerably smaller in length and still lead to the required webpage. 
# This is accomplished by means of an “HTTP Redirect” on a domain name that is short, which links to the webpage that has a long URL.

# If the URL is using Shortening Services, the value assigned to this feature is 1 (phishing) or else 0 (legitimate).
#listing shortening services

shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"

def tinyURL(url):
    match = re.search(shortening_services,url)
    if match:
        return 1
    else:
        return 0

#### 4.1.6. Preffix or Suffix "-" in Domain
# Phishing domains are often designed to trick users into believing they are legitimate websites, typically imitating well-known brands or organizations. 
# One common technique used in phishing domains is the inclusion of a hyphen ("-") to deceive users.

# The hyphen is sometimes inserted in a phishing domain to make it visually similar to the legitimate domain it is impersonating. 
# For example, if the legitimate domain is "example.com," the phishing domain may appear as "ex-ample.com" or "examp-le.com." 
# By inserting the hyphen, scammers hope to exploit users' inattentiveness or typosquatting, where users mistakenly type the wrong characters or misspell a legitimate domain.

def hyphenURL(url):
    if '-' in urlparse(url).netloc:
        return 1
    else:
        return 0

#### 4.1.7. Preffix or Suffix "_" in Domain
# Phishers often try to mimic legitimate websites by using domain names that are similar to the targeted website's domain. 
# They may use techniques such as replacing letters with numbers or using similar-looking characters. 
# However, the use of "_" is not typically used as a deliberate tactic to deceive users, as it is not a common practice in legitimate domain names.

def underscoreURL(url):
    if '_' in urlparse(url).netloc:
        return 1
    else:
        return 0

#### 4.1.8. Length of Sub-domain
# Phishing URLs may have excessively long subdomains or subdomain strings that appear unusual or unrelated to the legitimate domain. 
# For example, "login.yourbank.com.phishingsite.com."

def subURl(url):
    p = urlparse(url).path.split('.')
    sub = 0
    for j in range(len(p)):
        if len(p[j])!=0:
            sub = sub+1
    return sub

#### Keywords in URL

# Look for keywords in the URL that suggest fraudulent activities, such as "login", "client", "admin", "server", or "account." 
# Phishing URLs often try to deceive users into believing they are accessing sensitive pages.
#### 4.1.9. "Client" in string

def clientURL(url):
    if "client" in url:
        return 1
    else:
        return 0

#### 4.1.10. "Admin" in string
def adminURL(url):
    if "admin"in url:
        return 1
    else:
        return 0

#### 4.1.11. "Server" in string
def serverURL(url):
    if "server" in url:
        return 1
    else:
        return 0

#### 4.1.12. "Login" in string
def loginURL(url):
    if "login" in url:
        return 1
    else:
        return 0

### 4.2. Domain-based features
# - DNS record
# - Age of domain
# - End period of domain

# importing required packages for this section
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime

#### 4.2.1. DNS record
# For phishing websites, either the claimed identity is not recognized by the WHOIS database or no records founded for the hostname. 
# If the DNS record is empty or not found then, the value assigned to this feature is 1 (phishing) or else 0 (legitimate).
# DNS Record availability (DNS_Record)
# obtained in the featureExtraction function itself

#### 4.2.2. Age of domain
# This feature can be extracted from WHOIS database. Most phishing websites live for a short period of time. 
# The minimum age of the legitimate domain is considered to be 12 months for this project. 
# Age here is nothing but different between creation and expiration time.
# If age of domain > 12 months, the vlaue of this feature is 1 (phishing) else 0 (legitimate).

import whois
from datetime import datetime

def domainAge(domain_name):
    creation_date = domain_name.creation_date
    expiration_date = domain_name.expiration_date
    if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
        try:
            creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
            expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
        except:
            return 1
    if ((expiration_date is None) or (creation_date is None)):
        return 1
    elif ((type(expiration_date) is list) or (type(creation_date) is list)):
        return 1
    else:
        ageofdomain = abs((expiration_date - creation_date).days)
        if ((ageofdomain/30) < 6):
            age = 1
        else:
            age = 0
    return age

#### 4.2.3. End period of Domain
# This feature can be extracted from WHOIS database. For this feature, the remaining domain time is calculated by finding the different between expiration time & current time. 
# The end period considered for the legitimate domain is 6 months or less for this project.
# If end period of domain > 6 months, the vlaue of this feature is 1 (phishing) else 0 (legitimate).

def domainEnd(domain_name):
    expiration_date = domain_name.expiration_date
    if isinstance(expiration_date,str):
        try:
            expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
        except:
            return 1
    if (expiration_date is None):
        return 1
    elif (type(expiration_date) is list):
        return 1
    else:
        today = datetime.now()
        end = abs((expiration_date - today).days)
        if ((end/30) < 6):
            end = 0
        else:
            end = 1
    return end

### 4.3. Content-based Features
# - IFrame redirection
# - Webiste forwarding

#### 4.3.1. IFrame redirection
# IFrame is an HTML tag used to display an additional webpage into one that is currently shown. 
# Phishers can make use of the “iframe” tag and make it invisible i.e. without frame borders. 
# In this regard, phishers make use of the “frameBorder” attribute which causes the browser to render a visual delineation.

# If the iframe is empty or repsonse is not found then, the value assigned to this feature is 1 (phishing) or else 0 (legitimate).

import requests
import re

def iframe(response):
    if response == "":
        return 1
    else:
        if re.findall(r"[|]", response.text):
            return 0
        else:
            return 1

#### 4.3.2. Website forwarding
# The fine line that distinguishes phishing websites from legitimate ones is how many times a website has been redirected. 
# In our dataset, we find that legitimate websites have been redirected one time max. 
# On the other hand, phishing websites containing this feature have been redirected at least 4 times.

def forwarding(response):
    if response == "":
        return 1
    else:
        if len(response.history) <= 2:
            return 0
        else:
            return 1

## 5. URL feature extraction
# Create a list and a function that calls the other functions and stores all the features of the URL in the list. 
# We will extract the features of each URL and append to this list.

def featureExtraction(url, label):
    features = []

    #URL-based feature
    features.append(getDomain(url))
    features.append(lengthURL(url))
    features.append(depthURL(url))
    features.append(httpDomain(url))
    features.append(tinyURL(url))
    features.append(hyphenURL(url))
    features.append(underscoreURL(url))
    features.append(subURl(url))
    features.append(clientURL(url))
    features.append(adminURL(url))
    features.append(serverURL(url))
    features.append(loginURL(url))
        
    #Domain based features
    dns = 0
    try:
        domain_name = whois.whois(urlparse(url).netloc)
    except:
        dns = 1

    features.append(dns)
    features.append(1 if dns == 1 else domainAge(domain_name))
    features.append(1 if dns == 1 else domainAge(domain_name))

    #Content-based feature

    try:
        response = requests.get(url)
    except:
        response = ""
    
    features.append(iframe(response))
    features.append(forwarding(response))
    features.append(label)

    return features
    
### 5.1. Legitimate URLs:

# Feature extraction is done on legitimate URLs.
#Extracting the features & storing them in a list
legi_features = []
label = 0 #label = 0 (legitimate label)

for i in range(0, 1000):
    url = legi_url["url"][i]
    legi_features.append(featureExtraction(url,label))
    # print(legi_features)
#Converting the list to dataframe
feature_name = ["domain","url_len","url_dep","httpDomain","tinyURL","hyphenURL",
                "underscoreURL","subURl","clientURL","adminURL","serverURL","loginURL", "dns","domainAge","domainEnd","iframe","forwarding","label"]

legitimate = pd.DataFrame(legi_features, columns=feature_name)
legitimate.head()
legitimate.to_csv('data/legitimate.csv') #saving dataframe to csv

### 5.2. Phishing URLs:

# Feature extraction is done on phishing URLs.
#Extracting the features & storing them in a list

phish_features = []
label = 1 #label = 1 (phishing label)

for i in range(0, 1000):
    url = phish_url["url"][i]
    phish_features.append(featureExtraction(url,label))
    # print(legi_features)
#Converting the list to dataframe
feature_name = ["domain","url_len","url_dep","httpDomain","tinyURL","hyphenURL",
                "underscoreURL","subURl","clientURL","adminURL","serverURL","loginURL", "dns","domainAge","domainEnd","iframe","forwarding","label"]

phishing = pd.DataFrame(phish_features, columns=feature_name)
phishing.head()
phishing.tail()
phishing.to_csv('data/final_phishing.csv') #saving dataframe to csv

### 5.3. Final dataset
# In the above section we formed two dataframes of legitimate & phishing URL features. 
# Now, we will combine them to a single dataframe and export the data to csv file for the Machine Learning training done in other notebook
#Concatenating the dataframes into one 

urldata = pd.concat([legitimate, phishing]).reset_index(drop=True)
urldata = urldata.sample(frac=1, random_state=1).reset_index(drop=True)
urldata.head()
urldata.tail()
urldata.to_csv('data/urldata.csv', index=False) #saving final datadrame to csv

## 6. Loading final datastet
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
urldata = pd.read_csv("data/urldata.csv")
urldata.head()
urldata.tail()

## 7. Data Preprocessing and EDA of the URL dataset
### 7.1. Shape of the dataset
urldata.shape

### 7.2. Dataset Columns
urldata.columns

### 7.3. Dataset information
urldata.info()

### 7.4. Dataset Summary
urldata.describe
# The above obtained result shows that the most of the data is made of 0's & 1's except 'Domain' & 'URL_Depth' columns. 
# The Domain column doesnt have any significance to the machine learning model training. So dropping the 'Domain' column from the dataset.

### 7.5.Check of Null values
urldata.isnull().sum()
# In the feature extraction file, the extracted features of legitmate & phishing url datasets are just concatenated without any shuffling. 
# This resulted in top 1000 rows of legitimate url data & bottom 1000 of phishing url data.

### 7.6. Dropping column name "domain"
urldata = urldata.drop(['domain'],axis = 1)
urldata.columns

### 7.7. Max and Min of "url_len"
print("Maximum of url length::",urldata["url_len"].max())
print("Minimum of url length::",urldata["url_len"].min())

### 7.8. Max and Min of "url_depth"
print("Maximum of url depgth::",urldata["url_dep"].max())
print("Minimum of url length::",urldata["url_dep"].min())

## 8. Data Visualization 
### 8.1. Plotting the data distribution
urldata.hist(bins = 50,figsize = (15,15))
plt.show()
### 8.2. Correlation heatmap
plt.figure(figsize=(15,13))
sns.heatmap(urldata.corr())
plt.show()

## 9. Splitting the Dataset
y = urldata['label']
X = urldata.drop('label',axis=1)
X.shape, y.shape

# Splitting the dataset into train and test sets: 70-30 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, random_state = 12)
X_train.shape, X_test.shape

## 10. Machine Learning models and Training
# This data set comes under classification problem, as the input URL is classified as phishing (1) or legitimate (0). The supervised machine learning models (classification) considered to train the dataset:

# - Decision Tree
# - Random Forest
# - XGBoost
# - Logistic Regression
# - Support Vectore Machines (SVM)
# - Multilayer perceptrons

#importing packages
from sklearn.metrics import accuracy_score
# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

#function to call for storing the results
def storeResults(model, a,b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))

### 10.1. Decision Tree Classifier
# Decision trees are widely used models for classification and regression tasks. Essentially, they learn a hierarchy of if/else questions, leading to a decision. Learning a decision tree means learning the sequence of if/else questions that gets us to the true answer most quickly.

# In the machine learning setting, these questions are called tests (not to be confused with the test set, which is the data we use to test to see how generalizable our model is). 
# To build a tree, the algorithm searches over all possible tests and finds the one that is most informative about the target variable.

# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier

# instantiate the model 
tree = DecisionTreeClassifier(max_depth = 5)
# fit the model 
tree.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)
#### Performance Evaluation
#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

#### Storing Results:
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Decision Tree', acc_train_tree, acc_test_tree)

### 10.2. Random Forest
# Random forests for regression and classification are currently among the most widely used machine learning methods.
# A random forest is essentially a collection of decision trees, where each tree is slightly different from the others. The idea behind random forests is that each tree might do a relatively good job of predicting, but will likely overfit on part of the data.

# If we build many trees, all of which work well and overfit in different ways, we can reduce the amount of overfitting by averaging their results. 
# To build a random forest model, you need to decide on the number of trees to build (the n_estimators parameter of RandomForestRegressor or RandomForestClassifier). They are very powerful, often work well without heavy tuning of the parameters, and don’t require scaling of the data.
# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# instantiate the model
forest = RandomForestClassifier(max_depth=5)

# fit the model 
forest.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)
#### Performance Evaluation
#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))
#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
#### Storing Results
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Random Forest', acc_train_forest, acc_test_forest)

### 10.3. XGBoost
# XGBoost is one of the most popular machine learning algorithms these days. XGBoost stands for eXtreme Gradient Boosting. 
# Regardless of the type of prediction task at hand; regression or classification. 
# XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
#XGBoost Classification model
from xgboost import XGBClassifier

# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)
#### Performance Evaluation
#computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)

print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))
#### Storing Results
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('XGBoost', acc_train_xgb, acc_test_xgb)

### 10.4. Logistic Regression
# In machine learning, logistic regression is a classification algorithm used to model the relationship between input features and a binary output variable. It estimates the probability of the output belonging to a specific class using a logistic or sigmoid function.
# During training, it learns optimal parameters by minimizing a cost function, such as cross-entropy loss. Once trained, it can make predictions on new data by calculating the probability of belonging to a class. Logistic regression is widely used in various domains for binary classification tasks when a linear relationship is assumed.
#Logistic regression machine learning

from sklearn.linear_model import LogisticRegression

#instantive the model
logistic = LogisticRegression(random_state = 12)
#fit the model
logistic.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_log = logistic.predict(X_test)
y_train_log = logistic.predict(X_train)
#### Performance Evaluation
#computing the accuracy of the model performance
acc_train_log = accuracy_score(y_train,y_train_log)
acc_test_log = accuracy_score(y_test,y_test_log)

print("Logistic: Accuracy on training Data: {:.3f}".format(acc_train_log))
print("Logistic : Accuracy on test Data: {:.3f}".format(acc_test_log))
#### Storing Result
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Logistic', acc_train_log, acc_test_log)
### 10.5. Support Vector Machines
# In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 
# Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.
#Support vector machine model
from sklearn.svm import SVC

# instantiate the model
svm = SVC(kernel='linear', C=1.0, random_state=12)
#fit the model
svm.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)
#### Performance Evaluation
#computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train,y_train_svm)
acc_test_svm = accuracy_score(y_test,y_test_svm)

print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))
#### Storing Result
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('SVM', acc_train_svm, acc_test_svm)

### 10.6. Multilayer Perceptrons (MLPs)
# Multilayer perceptrons (MLPs) are also known as (vanilla) feed-forward neural networks, or sometimes just neural networks. 
# Multilayer perceptrons can be applied for both classification and regression problems.
# MLPs can be viewed as generalizations of linear models that perform multiple stages of processing to come to a decision.
# Multilayer Perceptrons model
from sklearn.neural_network import MLPClassifier

# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))

# fit the model 
mlp.fit(X_train, y_train)
#predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)
#### Performance Evaluation
#computing the accuracy of the model performance
acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))
#### Storing Result
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

## 11. Comparison of Models
# To compare the models performance, a dataframe is created. The columns of this dataframe are the lists created to store the results of the model.
#creating dataframe
results = pd.DataFrame({ 'ML Model': ML_Model,    
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results
#Sorting the datafram on accuracy
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
# For the above comparision, it is clear that the XGBoost Classifier works well with this dataset with 98.7% accuracy.

# So, saving the model for future use.
# save XGBoost model to file

import pickle
pickle.dump(xgb, open("/kaggle/working/XGBoostClassifier.pickle.dat", "wb"))
### Testing the saved model:
# load model from file
loaded_model = pickle.load(open("data/XGBoostClassifier.pickle.dat", "rb"))
loaded_model