# importing required packages for this section
from urllib.parse import urlparse,urlencode
import ipaddress
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime
import requests
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

pickel_file = "data\XGBoostClassifier.pkl"


# Length of URL

def lengthURL(url):
    return len(url)

# Depth of URL

def depthURL(url):
    s = urlparse(url).path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j])!=0:
            depth = depth+1
    return depth


# "http/https" in Domain name

def httpDomain(url):
    if "https" in url:
        return 1
    else:
        return 0

# URL shortening

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
    

# Preffix or Suffix "-" in Domain

def hyphenURL(url):
    if '-' in urlparse(url).netloc:
        return 1
    else:
        return 0

#### 4.1.7. Preffix or Suffix "_" in Domain

def underscoreURL(url):
    if '_' in urlparse(url).netloc:
        return 1
    else:
        return 0

# Length of Sub-domain

def subURl(url):
    p = urlparse(url).path.split('.')
    sub = 0
    for j in range(len(p)):
        if len(p[j])!=0:
            sub = sub+1
    return sub

# "Client" in string

def clientURL(url):
    if "client" in url:
        return 1
    else:
        return 0
    
# "Admin" in string

def adminURL(url):
    if "admin"in url:
        return 1
    else:
        return 0
    
# "Server" in string
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

# DNS record
# For phishing websites, either the claimed identity is not recognized by the WHOIS database or no records founded for the hostname. If the DNS record is empty or not found then, the value assigned to this feature is 1 (phishing) or else 0 (legitimate).
# DNS Record availability (DNS_Record)
# obtained in the featureExtraction function itself


# Age of domain

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

# End period of Domain

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


# IFrame redirection

def iframe(response):
    if response == "":
        return 1
    else:
        if re.findall(r"[|]", response.text):
            return 0
        else:
            return 1
        
# Website forwarding

def forwarding(response):
    if response == "":
        return 1
    else:
        if len(response.history) <= 2:
            return 0
        else:
            return 1
        
def featureExtraction(url):
    features = []

    #URL-based feature
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

    return features

@app.route('/')
def home():
    return render_template('home.html')         


@app.route('/predict', methods=['POST'])
def predict():
    # Fetch the input text.
    url = request.form['url']
    print(url)

    # Feature Extraction of the URL
    fe = np.array([featureExtraction(url)])

    # Loading model
    model = pickle.load(open('data/XGBoostClassifier.pkl', 'rb'))

        # Model Prediction

    pred = model.predict(fe)

    if pred[0] == 1:
        result = "Phishing Website"
    else:
        result = "Legitimate Website"
    return render_template('after.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)

