#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# ### Scraping Neighborhoods data of Toronto from Wikipedia

# In[37]:


import bs4 as bs
import pickle
import requests


# In[44]:


def save_toronto_data():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    ticker1 = []
    ticker2 = []
    ticker3 = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker1.append(ticker)
        ticker = row.findAll('td')[1].text
        ticker2.append(ticker)
        ticker = row.findAll('td')[2].text
        ticker3.append(ticker)
    
    tickers = [ticker1, ticker2, ticker3]
        
    with open("torontotickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return ticker1, ticker2, ticker3

toronto_data = save_toronto_data()
toronto_data


# ### Change the list into dataframe and delete symbol '\n' from each element in the list 'Neighborhoods'

# In[61]:


data_tor = list(map(list, zip(*list(toronto_data))))
df = pd.DataFrame(data_tor)
df.columns = ['PostalCode', 'Borough', 'Neighborhoods']
df['Neighborhoods'] = df['Neighborhoods'].str.strip('\n')
df


# ### Ignore cells with a borough that is Not assigned.

# In[66]:


df.drop(df[df.Borough == 'Not assigned'].index, inplace=True)


# ### Assign elements which are 'Not assigned' in the list 'Neighborhoods' with contents that are the same with the ones in list 'Borough' from the same row.

# In[84]:


for index,row in df.iterrows():
    if row['Neighborhoods'] == 'Not assigned':
        row['Neighborhoods'] = row['Borough']


# ### Merge rows which have same postalcode with the neighborhoods separated with a comma.

# In[83]:


df.groupby('PostalCode')['Neighborhoods'].apply(','.join).reset_index()


# In[85]:


df.shape

