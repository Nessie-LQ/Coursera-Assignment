#!/usr/bin/env python
# coding: utf-8

# # Coursera capstone

# #### Name: Qing Li
# #### Date: 27/10/2019

# ## Table of contents:
# - [Introduction: Background and Business Problem](#introduction)
# - [Data Description](#data)
# - [Methodology](#methodology)
# - [Analysis](#analysis)
# - [Results and Discussion](#results)
# - [Conclusion](#conclusion)

# ## Introduction: Background and Business Problem<a name="introduction"></a>

# As we know, Shanghai is a  promising international city, which is the financial and trade center of China. During recent years, a large number of companies, including both those mature ones who have strong backgrounds and strat-up companies, perfer to open a branch in Shanghai. This project aims at finding an optimal location for stakeholders interested in starting a company.
# 
# Since this is a start-up company, I am going to **avoid neighborhoods which located too closed to the city center** because of the high rent of offices. However, locating too far away form the business center of the city brings no advantages to the company and employees. Therefore, the location will be set in areas which are certainly distant from the city center. Besides, the purpose of this project is to choose a location for a company, so the places of interest are restricted into the category called coworking spaces. Also, locations with **multiple restaurants around**, which are convenient for employees to have lunch everyday, will be considered preferentially. Besides,  considering  of commuting factors, I would also like to choose locations which **have metro stations nearby**.
# 
# Based on this criteria, I am going to create a map and information charts to show the promising locations and their advantages.

# ## Data Description<a name="data"></a>

# To consider about our business problem, we can list the factors that will influence our data:
# 
# - distance of neighborhood from the city center (not too closed)
# - number of coworking spaces in the neighborhood
# - number of existing restaurants in the neighborhood (any type of restaurant)
# - distance of neighborhood from neareast metro station
# 
# Based on these limitations, I decide to use regularly spaced grid of locations, centered around 3-10km from city center, to define our neighborhoods.
# 
# Datas we used will be extracted from following resources:
# 
# - Google Maps API reverse geocoding will be used to generate the addresses of centers of candidate areas
# - number of restaurants and their per capita consumption and location in every neighborhood will be obtained using Foursquare API
# - coordinate of Shanghai center will be obtained using Google Maps API geocoding of well known Shanghai location (Huangpu, which is the most bustling area in Shanghai).

# ## Methodology<a name="methodology"></a>

# ### Neighborhood Candidates

# For the first step, I create latitude & longitude coordinates for centroids of the candidate neighborhoods. I will create a grid of cells covering the area of interest which is aprox. 3x3-12x12 killometers away and centered around Shanghai city center (Huangpu).
# 
# Now, Google Maps geocoding API is used to generate the latitude & longtitude of Shanghai city center.

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# import k-means from clustering stage
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import seaborn as sns

import folium # map rendering library

import pickle

print('Libraries imported.')


# In[2]:


def get_coordinates(api_key, address, verbose=False):
    try:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(api_key, address)
        response = requests.get(url).json()
        if verbose:
            print('Google Maps API JSON result =>', response)
        results = response['results']
        geographical_data = results[0]['geometry']['location'] # get geographical coordinates
        lat = geographical_data['lat']
        lon = geographical_data['lng']
        return [lat, lon]
    except:
        return [None, None]
    
address = 'Huangpu, Shanghai, China'
google_api_key = 'AIzaSyCxrGdn5VOxLZ8G8h41KlVuf7_QSflfL80'
shanghai_center = get_coordinates(google_api_key, address)
print('Coordinate of {}: {}'.format(address, shanghai_center))


# Next, I will create a grid of area candidates, equaly spaced, 3x3-10x10 killometers away and centered around city center, Huangpu District. Our neighborhoods will be defined as circular areas with a radius of 300 meters, so our neighborhood centers will be 600 meters apart.
# 
# To accurately calculate distances, a grid of locations in Cartesian 2D coordinate system which allows us to calculate distances in meters (not in latitude/longitude degrees) must be created first. Then those coordinates will be projected back to latitude/longitude degrees to be shown on Folium map. 
# 
# Now, I am going to create functions to convert between WGS84 spherical coordinate system (latitude/longitude degrees) and UTM Cartesian coordinate system (X/Y coordinates in  meters).

# In[3]:


import shapely.geometry

import pyproj

import math

def lonlat_to_xy(lon, lat):
    proj_latlon = pyproj.Proj(proj='latlong',datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=51, datum='WGS84')
    xy = pyproj.transform(proj_latlon, proj_xy, lon, lat)
    return xy[0], xy[1]

def xy_to_lonlat(x, y):
    proj_latlon = pyproj.Proj(proj='latlong',datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=51, datum='WGS84')
    lonlat = pyproj.transform(proj_xy, proj_latlon, x, y)
    return lonlat[0], lonlat[1]

def calc_xy_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx*dx + dy*dy)

print('Coordinate transformation check')
print('-------------------------------')
print('Shanghai center longitude={}, latitude={}'.format(shanghai_center[1], shanghai_center[0]))
x, y = lonlat_to_xy(shanghai_center[1], shanghai_center[0])
print('Shanghai center UTM X={}, Y={}'.format(x, y))
lo, la = xy_to_lonlat(x, y)
print('Shanghai center longitude={}, latitude={}'.format(lo, la))


# Now, it's time to create a hexagonal grid of cells. The method is offseting every other row, then adjust vertical row spacing to make sure that every cell is equally distant from all of its neighbors.

# In[4]:


shanghai_center_x, shanghai_center_y = lonlat_to_xy(shanghai_center[1], shanghai_center[0]) # City center in Cartesian coordinates

k = math.sqrt(3) / 2 # Vertical offset for hexagonal grid cells
x_min = shanghai_center_x - 6000
x_step = 600
y_min = shanghai_center_y - 6000 - (int(21/k)*k*600-12000)/2
y_step = 600 * k 

latitudes = []
longitudes = []
distances_from_center = []
xs = []
ys = []
for i in range(0, int(21/k)):
    y = y_min + i * y_step
    x_offset = 300 if i%2==0 else 0
    for j in range(0, 21):
        x = x_min + j * x_step + x_offset
        distance_from_center = calc_xy_distance(shanghai_center_x, shanghai_center_y, x, y)
        if (3000 <= distance_from_center <= 10001):
            lon, lat = xy_to_lonlat(x, y)
            latitudes.append(lat)
            longitudes.append(lon)
            distances_from_center.append(distance_from_center)
            xs.append(x)
            ys.append(y)

print(len(latitudes), 'candidate neighborhood centers generated.')


# In[5]:


map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.Marker(shanghai_center, popup='Huangpu').add_to(map_shanghai)
for lat, lon in zip(latitudes, longitudes):
    folium.CircleMarker([lat, lon], radius=2, color='red', fill=True, fill_color='blue', fill_opacity=1).add_to(map_shanghai) 
    folium.Circle([lat, lon], radius=300, color='blue', fill=False).add_to(map_shanghai)
map_shanghai


# After evaluating the coordinates of centers of neighbors I choose, I am going to get the addresses of these centers by using Google Maps API

# In[6]:


def get_address(api_key, latitude, longitude, verbose=False):
    try:
        url = 'https://maps.googleapis.com/maps/api/geocode/json?key={}&latlng={},{}'.format(api_key, latitude, longitude)
        response = requests.get(url).json()
        if verbose:
            print('Google Maps API JSON result =>', response)
        results = response['results']
        address = results[0]['formatted_address']
        return address
    except:
        return None

addr = get_address(google_api_key, shanghai_center[0], shanghai_center[1])
print('Reverse geocoding check')
print('-----------------------')
print('Address of [{}, {}] is: {}'.format(shanghai_center[0], shanghai_center[1], addr))


# In[7]:


print('Obtaining location addresses: ', end='')
addresses = []
for lat, lon in zip(latitudes, longitudes):
    address = get_address(google_api_key, lat, lon)
    if address is None:
        address = 'NO ADDRESS'
    address = address.replace(', China', '') # We don't need country part of address
    addresses.append(address)
    print(' .', end='')
print(' done.')


# Now I have already got the addresses of these centers of chosen neighbors, which are shown in the next cell.

# In[8]:


addresses[0:10]


# In order to facilitate the processing of subsequent data, the addresses and their corresponding latitudes and longtitudes will be organized into a dataframe.

# In[9]:


df_locations = pd.DataFrame({'Address': addresses,
                             'Latitude': latitudes,
                             'Longitude': longitudes,
                             'X': xs,
                             'Y': ys,
                             'Distance from center': distances_from_center})

df_locations.head(10)


# ## Foursquare

# For now, I have already got the specific information of the neighborhoods meeting my needs initially. Then I am going to use Foursquare API to get info on restaurants and metro stations in each neighborhood for further comparison.

# Foursquare credentials are defined in hidden cell bellow.

# In[77]:


CLIENT_ID = 'CRUQT3HO25H1L5ZDIPUIJ3UVNN0XVHRR3GP0V0JR2ABAQ2GA'
CLIENT_SECRET = 'EERHQ0LYMWSD4R4V23O4DSDLOZYV4MB5YY2TBL1DKTJRGKTZ' 
VERSION = '20191031' 

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[33]:


def get_categories(categories):
    return [(cat['name'], cat['id']) for cat in categories]

def format_address(df):
    address = ', '.join(df['formattedAddress'])
    address = address.replace('China, ', '')
    address = address.replace('Shanghai, ', '')
    return address

def get_venues_near_location(lat, lon, category, client_id, client_secret, version, radius=350, limit=100):
    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&categoryId={}&radius={}&limit={}'.format(
        client_id, client_secret, version, lat, lon, category, radius, limit)
    try:
        results = requests.get(url).json()['response']['groups'][0]['items']
        venues = [(item['venue']['id'],
                   item['venue']['name'],
                   get_categories(item['venue']['categories']),
                   (item['venue']['location']['lat'], item['venue']['location']['lng']),
                   format_address(item['venue']['location']),
                   item['venue']['location']['distance']) for item in results]        
    except:
        venues = []
    return venues


# In[99]:


food_category = '4d4b7105d754a06374d81259'

restaurants = {}
location_restaurants = []

print('Obtaining venues around candidate locations:', end='')
for lat, lon in zip(latitudes, longitudes):
    # Using radius=350 to meke sure we have overlaps/full coverage so we don't miss any restaurant (we're using dictionaries to remove any duplicates resulting from area overlaps)
    venues = get_venues_near_location(lat, lon, food_category, CLIENT_ID, CLIENT_SECRET, VERSION, radius=350, limit=100)
    area_restaurants = []
    for venue in venues:
        venue_id = venue[0]
        venue_name = venue[1]
        venue_categories = venue[2]
        venue_latlon = venue[3]
        venue_address = venue[4]
        venue_distance = venue[5]
        x, y = lonlat_to_xy(venue_latlon[1], venue_latlon[0])
        restaurant = (venue_id, venue_name, venue_latlon[0], venue_latlon[1], venue_address, venue_distance, x, y)
        if venue_distance<=300:
            area_restaurants.append(restaurant)
        restaurants[venue_id] = restaurant
    location_restaurants.append(area_restaurants)
    print(' .', end='')
print(' done.')


# In[100]:


print('Total number of restaurants:', len(restaurants))
print('Average number of restaurants in neighborhood:', np.array([len(r) for r in location_restaurants]).mean())


# In[37]:


map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.Marker(shanghai_center, popup='Huangpu').add_to(map_shanghai)
for res in restaurants.values():
    lat = res[2]; lon = res[3]
    color = 'red'
    folium.CircleMarker([lat, lon], radius=3, color=color, fill=True, fill_color=color, fill_opacity=1).add_to(map_shanghai)
map_shanghai


# In[42]:


metrostation_category = '4bf58dd8d48988d1fd931735'

metrostations = {}
location_metro = []

print('Obtaining venues around candidate locations:', end='')
for lat, lon in zip(latitudes, longitudes):
    # Using radius=350 to meke sure we have overlaps/full coverage so we don't miss any restaurant (we're using dictionaries to remove any duplicates resulting from area overlaps)
    venues = get_venues_near_location(lat, lon, metrostation_category, CLIENT_ID, CLIENT_SECRET, VERSION, radius=350, limit=100)
    area_metrostations = []
    for venue in venues:
        venue_id = venue[0]
        venue_name = venue[1]
        venue_categories = venue[2]
        venue_latlon = venue[3]
        venue_address = venue[4]
        venue_distance = venue[5]
        x, y = lonlat_to_xy(venue_latlon[1], venue_latlon[0])
        metrostation = (venue_id, venue_name, venue_latlon[0], venue_latlon[1], venue_address, venue_distance, x, y)
        if venue_distance<=300:
            area_metrostations.append(metrostation)
        metrostations[venue_id] = metrostation
    location_metro.append(area_metrostations)
    print(' .', end='')
print(' done.')      


# In[43]:


print('Total number of metro stations:', len(metrostations))


# In[44]:


map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.Marker(shanghai_center, popup='Huangpu').add_to(map_shanghai)
for metro in metrostations.values():
    lat = metro[2]; lon = metro[3]
    color = 'green'
    folium.CircleMarker([lat, lon], radius=3, color=color, fill=True, fill_color=color, fill_opacity=1).add_to(map_shanghai)
map_shanghai


# Since a start-up company may prefer to choose office in an office building, the last filtering will take place in the category called coworking space.

# In[122]:


coworking_category = '4bf58dd8d48988d174941735'
coworkspaces = {}
location_cowork = []

print('Obtaining venues around candidate locations:', end='')
for lat, lon in zip(latitudes, longitudes):
    # Using radius=350 to meke sure we have overlaps/full coverage so we don't miss any restaurant (we're using dictionaries to remove any duplicates resulting from area overlaps)
    venues = get_venues_near_location(lat, lon, coworking_category, CLIENT_ID, CLIENT_SECRET, VERSION, radius=350, limit=100)
    area_coworkspaces = []
    for venue in venues:
        venue_id = venue[0]
        venue_name = venue[1]
        venue_categories = venue[2]
        venue_latlon = venue[3]
        venue_address = venue[4]
        venue_distance = venue[5]
        x, y = lonlat_to_xy(venue_latlon[1], venue_latlon[0])
        coworkspace = (venue_id, venue_name, venue_latlon[0], venue_latlon[1], venue_address, venue_distance, x, y)
        if venue_distance<=300:
            area_coworkspaces.append(coworkspace)
        coworkspaces[venue_id] = coworkspace
    location_cowork.append(area_coworkspaces)
    print(' .', end='')
print(' done.')
    


# In[124]:


map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.Marker(shanghai_center, popup='Huangpu').add_to(map_shanghai)
for cowork in coworkspaces.values():
    lat = cowork[2]; lon = cowork[3]
    color = 'orange'
    folium.CircleMarker([lat, lon], radius=3, color=color, fill=True, fill_color=color, fill_opacity=1).add_to(map_shanghai)
map_shanghai


# ## Analysis<a name="analysis"></a>

# After getting the basic information which is needed for the further analysis. I am going to derive some hidden info from the raw data. Firstly, all the raw data will be aggregate into a dataframe.

# In[125]:


location_restaurants_count = [len(res) for res in location_restaurants]

df_locations['Restaurants in area'] = location_restaurants_count

print('Average number of restaurants in every area with radius=300m:', np.array(location_restaurants_count).mean())

df_locations.head(10)


# In[126]:


location_cowork_count = [len(cow) for cow in location_cowork]

df_locations['Coworking spaces in area'] = location_cowork_count

print('Average number of coworking spaces in every area with radius=300m:', np.array(location_cowork_count).mean())

df_locations.head(10)


# In[127]:


location_metro_count = [len(res) for res in location_metro]

df_locations['Metro stations in area'] = location_metro_count

print('Average number of metro stations in every area with radius=300m:', np.array(location_metro_count).mean())

df_locations.head(10)


# In[128]:


distances_to_nearest_metro = []

for area_x, area_y in zip(xs, ys):
    min_distance = 10000
    for metr in metrostations.values():
        metr_x = metr[6]
        metr_y = metr[7]
        d = calc_xy_distance(area_x, area_y, metr_x, metr_y)
        if d<min_distance:
            min_distance = d
    distances_to_nearest_metro.append(min_distance)

df_locations['Distance to the nearest metro station'] = distances_to_nearest_metro


# In[129]:


df_locations.head(10)


# Finding out those neighborhoods which have metro stations and at least one restaurant in the area.

# In[130]:


df_have_metro = df_locations.drop(df_locations[df_locations['Metro stations in area'] == 0].index)
df_have_resmetro = df_have_metro.drop(df_have_metro[df_have_metro['Restaurants in area'] == 0].index)
df_select = df_have_resmetro.drop(df_have_resmetro[df_have_resmetro['Coworking spaces in area'] == 0].index)
df_select


# After filtering the dataframe into a new one, df_selected, the confusing row index needs to be reset.

# In[131]:


df_select.reset_index(drop=True, inplace=True)


# In[132]:


def formatted_address(df):
    for i in range(len(df)):
        address = df.loc[i, 'Address']
        address = address.replace('China, ', '')
        address = address.replace('Shanghai, ', '')
        df.loc[i, 'Address'] = address
    return df

df_select = formatted_address(df_select)
df_select


# In[137]:


plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

df_barplt = pd.Series(df_select['Metro stations in area'].values, index = df_select['Address'])
df_barplt.plot(kind = 'bar', figsize = (18,10))

plt.xlabel('Neighborhoods') # add to x-label to the plot
plt.ylabel('Number of metro stations') # add y-label to the plot
plt.title('Number of metro stations in the area of each neighborhood') # add title to the plot

plt.show()


# In[138]:


df_barplt = pd.Series(df_select['Restaurants in area'].values, index = df_select['Address'])
df_barplt.plot(kind = 'bar', figsize = (18,10), color = 'red')

plt.xlabel('Neighborhoods') # add to x-label to the plot
plt.ylabel('Number of restaurants') # add y-label to the plot
plt.title('Number of restuarants in the area of each neighborhood') # add title to the plot

plt.show()


# In[144]:


df_barplt = pd.Series(df_select['Coworking spaces in area'].values, index = df_select['Address'])
df_barplt.plot(kind = 'bar', figsize = (18,10), color = 'green')

plt.xlabel('Neighborhoods') # add to x-label to the plot
plt.ylabel('Number of coworking spaces') # add y-label to the plot
plt.title('Number of coworking spaces in the area of each neighborhood') # add title to the plot

plt.show()


# Since the company may prefer the locations that have metro station situated as near as possible. The dataframe df_select will be reformed by following the rule of ascending 'Distance to the nearest metro station'. 

# In[139]:


df_select = df_select.sort_values(by="Distance to the nearest metro station" , ascending = True)
df_select


# Now, I will create a heatmap to see the distributions of restuarants and metro stations in neighborhoods that I select more clearly.

# In[176]:


with open(r"./shanghai.json", encoding='utf-8', errors='ignore') as json_data:
     shanghai_geo = json.load(json_data, strict=False)

def boroughs_style(feature):
    return { 'color': 'blue', 'fill': False }

restaurant_latlons = [[res[2], res[3]] for res in restaurants.values()]

from folium import plugins
from folium.plugins import HeatMap

map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.TileLayer('cartodbpositron').add_to(map_shanghai) #cartodbpositron cartodbdark_matter
HeatMap(restaurant_latlons).add_to(map_shanghai)
folium.Marker(shanghai_center).add_to(map_shanghai)
folium.Circle(shanghai_center, radius=3000, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=6500, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=10000, fill=False, color='white').add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# In[177]:


metro_latlons = [[metr[2], metr[3]] for metr in metrostations.values()]

map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.TileLayer('cartodbpositron').add_to(map_shanghai) #cartodbpositron cartodbdark_matter
HeatMap(metro_latlons).add_to(map_shanghai)
folium.Marker(shanghai_center).add_to(map_shanghai)
folium.Circle(shanghai_center, radius=3000, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=6500, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=10000, fill=False, color='white').add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# In[178]:


cowork_latlons = [[cow[2], cow[3]] for cow in coworkspaces.values()]

map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.TileLayer('cartodbpositron').add_to(map_shanghai) #cartodbpositron cartodbdark_matter
HeatMap(cowork_latlons).add_to(map_shanghai)
folium.Marker(shanghai_center).add_to(map_shanghai)
folium.Circle(shanghai_center, radius=3000, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=6500, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=10000, fill=False, color='white').add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# In[389]:


select_latitudes = df_select['Latitude'].values
select_longitudes = df_select['Longitude'].values

select_latlons = [[lat, lon] for lat, lon in zip(select_latitudes, select_longitudes)]

map_shanghai = folium.Map(location=shanghai_center, zoom_start=13)
folium.TileLayer('cartodbpositron').add_to(map_shanghai) #cartodbpositron cartodbdark_matter
HeatMap(select_latlons).add_to(map_shanghai)
folium.Marker(shanghai_center).add_to(map_shanghai)
folium.Circle(shanghai_center, radius=3000, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=6500, fill=False, color='white').add_to(map_shanghai)
folium.Circle(shanghai_center, radius=10000, fill=False, color='white').add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# As we can see, high density of restaurants, metro stations and coworking spaces can be found south-west from the city center of Shanghai. What's more important is that the locations survived from preliminary screening are also concentrated in the south-west from city center of Shanghai. By considering the all three factors, the focus of further analysis in the next stage will around south-west, which is the area that consists of Jingan District, Xuhui District and a part of Changning District.

# The three Districts that I choose as potential locations for office are famous for providing convenience at aspects of business, technology and traffic respectively. Jingan District is known as "High-grade commercial business district", Xuhui District is known as "Science and Education Cultural Core Area of Shanghai" and Changning District is the earliest and most mature district for business, which is also very convenient in traffic. 
# 
# All the three districts have their individual advantages. Firstly, I am going to narrow down the region of interesting locations to the high-coworkingspaces-density neighborhoods in Jingan District, Xuhui District and Changning District which meet the location requirements set initially.

# In[376]:


roi_x_min = shanghai_center_x - 6600
roi_y_max = shanghai_center_y + 1000
roi_width = 7000
roi_height = 7000
roi_center_x = roi_x_min + 3500
roi_center_y = roi_y_max - 3500
roi_center_lon, roi_center_lat = xy_to_lonlat(roi_center_x, roi_center_y)
roi_center = [roi_center_lat, roi_center_lon]

map_shanghai = folium.Map(location=roi_center, zoom_start=14)
HeatMap(cowork_latlons).add_to(map_shanghai)
folium.Circle(roi_center, radius=3500, color='white', fill=True, fill_opacity=0.4).add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# Now, we have got the new interesting region. The next step is to create some new, more dense location candidates according to the narrow-down region. The candidates will be set 150m apart.

# In[377]:


k = math.sqrt(3) / 2 # Vertical offset for hexagonal grid cells
x_step = 150
y_step = 150 * k 
roi_y_min = roi_center_y - 3500

roi_latitudes = []
roi_longitudes = []
roi_xs = []
roi_ys = []
for i in range(0, int(71/k)):
    y = roi_y_min + i * y_step
    x_offset = 75 if i%2==0 else 0
    for j in range(0, 71):
        x = roi_x_min + j * x_step + x_offset
        d = calc_xy_distance(roi_center_x, roi_center_y, x, y)
        if (d <= 3501):
            lon, lat = xy_to_lonlat(x, y)
            roi_latitudes.append(lat)
            roi_longitudes.append(lon)
            roi_xs.append(x)
            roi_ys.append(y)

print(len(roi_latitudes), 'candidate neighborhood centers generated.')


# In[378]:


def count_restaurants_nearby(x, y, restaurants, radius=250):    
    count = 0
    for res in restaurants.values():
        res_x = res[6]; res_y = res[7]
        d = calc_xy_distance(x, y, res_x, res_y)
        if d<=radius:
            count += 1
    return count

def count_cowork_nearby(x, y, coworkspaces, radius=250):    
    count = 0
    for cow in coworkspaces.values():
        cow_x = cow[6]; cow_y = cow[7]
        d = calc_xy_distance(x, y, cow_x, cow_y)
        if d<=radius:
            count += 1
    return count

def find_nearest_metro(x, y, metrostations):
    d_min = 100000
    for metr in metrostations.values():
        metr_x = metr[6]; metr_y = metr[7]
        d = calc_xy_distance(x, y, metr_x, metr_y)
        if d<=d_min:
            d_min = d
    return d_min

roi_restaurant_counts = []
roi_cowork_counts = []
roi_metro_distances = []

print('Generating data on location candidates... ', end='')
for x, y in zip(roi_xs, roi_ys):
    count1 = count_restaurants_nearby(x, y, restaurants, radius=250)
    roi_restaurant_counts.append(count1)
    count2 = count_cowork_nearby(x, y, coworkspaces, radius=250)
    roi_cowork_counts.append(count2)
    distance = find_nearest_metro(x, y, metrostations)
    roi_metro_distances.append(distance)
print('done.')


# In[379]:


df_roi_locations = pd.DataFrame({'Latitude':roi_latitudes,
                                 'Longitude':roi_longitudes,
                                 'X':roi_xs,
                                 'Y':roi_ys,
                                 'Restaurants nearby':roi_restaurant_counts,
                                 'Coworking spaces nearby':roi_cowork_counts,
                                 'Distance to nearest metro station':roi_metro_distances})

df_roi_locations.head(10)


# In[381]:


good_latitudes = df_roi_locations['Latitude'].values
good_longitudes = df_roi_locations['Longitude'].values

good_locations = [[lat, lon] for lat, lon in zip(good_latitudes, good_longitudes)]

map_shanghai = folium.Map(location=roi_center, zoom_start=14)
folium.TileLayer('cartodbpositron').add_to(map_shanghai)
HeatMap(cowork_latlons).add_to(map_shanghai)
folium.Circle(roi_center, radius=3500, color='white', fill=True, fill_opacity=0.6).add_to(map_shanghai)
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1, prefer_canvas=True).add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# Since only locations which have at least 3 restaurants, 1 coworking space in radius of 250 meters and metro stations in raidus of 300 meters are in consideration. I am going to filter these locations to extract ones meeting requirements.

# In[382]:


good_res_count = np.array((df_roi_locations['Restaurants nearby']>=3))
print('Locations with no more than two restaurants nearby:', good_res_count.sum())

good_cow_count = np.array((df_roi_locations['Coworking spaces nearby']>=1))
print('Locations with more than one coworking space nearby:', good_cow_count.sum())

good_metro_distance = np.array(df_roi_locations['Distance to nearest metro station']<=301)
print('Locations with at least one metro station within 300m:', good_metro_distance.sum())

good_locations = np.logical_and(good_res_count, good_cow_count, good_metro_distance)
print('Locations with both conditions met:', good_locations.sum())

df_good_locations = df_roi_locations[good_locations]


# In[383]:


good_latitudes = df_good_locations['Latitude'].values
good_longitudes = df_good_locations['Longitude'].values

good_locations = [[lat, lon] for lat, lon in zip(good_latitudes, good_longitudes)]

map_shanghai = folium.Map(location=roi_center, zoom_start=14)
folium.TileLayer('cartodbpositron').add_to(map_shanghai)
HeatMap(cowork_latlons).add_to(map_shanghai)
folium.Circle(roi_center, radius=3500, color='white', fill=True, fill_opacity=0.6).add_to(map_shanghai)
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1).add_to(map_shanghai) 
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# In[384]:


map_shanghai = folium.Map(location=roi_center, zoom_start=14)
HeatMap(good_locations, radius=25).add_to(map_shanghai)
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1).add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# As shown in the map above, we can see a clear indication of locations with at least 2 restaurants, 1 coworking space and 1 metro station nearby. Finally, these locations will be clustered into 15 segments and centers of each cluster will be generated to be the final results of this analysis.

# In[385]:


from sklearn.cluster import KMeans

number_of_clusters = 15

good_xys = df_good_locations[['X', 'Y']].values
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(good_xys)

cluster_centers = [xy_to_lonlat(cc[0], cc[1]) for cc in kmeans.cluster_centers_]

map_shanghai = folium.Map(location=roi_center, zoom_start=14)
folium.TileLayer('cartodbpositron').add_to(map_shanghai)
HeatMap(cowork_latlons).add_to(map_shanghai)
folium.Circle(roi_center, radius=2500, color='white', fill=True, fill_opacity=0.4).add_to(map_shanghai)
for lon, lat in cluster_centers:
    folium.Circle([lat, lon], radius=500, color='green', fill=True, fill_opacity=0.25).add_to(map_shanghai) 
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1).add_to(map_shanghai)
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# In[386]:


map_shanghai = folium.Map(location=roi_center, zoom_start=14)
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.Circle([lat, lon], radius=250, color='#00000000', fill=True, fill_color='#0066ff', fill_opacity=0.07).add_to(map_shanghai)
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=1).add_to(map_shanghai)
for lon, lat in cluster_centers:
    folium.Circle([lat, lon], radius=500, color='green', fill=False).add_to(map_shanghai) 
folium.GeoJson(shanghai_geo, style_function=boroughs_style, name='geojson').add_to(map_shanghai)
map_shanghai


# In[387]:


candidate_area_addresses = []
print('==============================================================')
print('Addresses of centers of areas recommended for further analysis')
print('==============================================================\n')
for lon, lat in cluster_centers:
    address = get_address(google_api_key, lat, lon).replace(', China', '')
    addr = address.replace(', Shanghai Shi', '')
    candidate_area_addresses.append(addr)    
    print('{}'.format(addr))


# In[388]:


map_shanghai = folium.Map(location=roi_center, zoom_start=14)
for lonlat, addr in zip(cluster_centers, candidate_area_addresses):
    folium.Marker([lonlat[1], lonlat[0]]).add_to(map_shanghai) 
for lat, lon in zip(good_latitudes, good_longitudes):
    folium.Circle([lat, lon], radius=250, color='#0000ff00', fill=True, fill_color='#0066ff', fill_opacity=0.05).add_to(map_shanghai)
map_shanghai


# ## Results and Discussion<a name="results"></a>

# Our analysis shows that although there are not too many coworking spaces(~227) located in our initial region of interest(3x3-12x12 killometers away and centered around Shanghai city center, Huangpu), high density of coworking spaces still can be found in Jingan District, Xuhui District and Changning District. Besides, in order to facilitate daily lives of employees, we set another two rules, number of restaurants nearby and distance to the nearest metro station, to narrow down the area of interest. By combining these two requirements, we can find that high density of potential locations with at least one coworking spaces, one restaurant and one metro station within the neighborhood at the south-west from city center of Shanghai. Therefore, our attention was then focused on Jingan District, a part of both Xuhui District and Changning District, which meet our requirements most and are famous for providing convenience at aspects of business, technology and traffic respectively.
# 
# After directing our attention to this more narrow area of interest(locating between 3km to 12km from center of Huangpu, covering approx. 7x7km south-west from Huangpu), we first created a dense grid of location candidates (spaced 150m appart); those locations were then filtered so that those with less than three restaurants and one coworking space in radius of 250m and those with no metro station in radius of 300m were removed. 
# 
# Those location candidates were then clustered to create 15 zones of interest which contain greatest number of location candidates using K means method. Addresses of centers of those zones were also generated using reverse geocoding to be used as markers/starting points for more detailed local analysis based on other factors.
# 
# These 15 zones as results containing largest number of potential start-up company locations based on coworking spaces, restaurants nearby and distance to metro stations. However, this result does not mean these zones are absolutely the optimal locations for start-up companies. Since Shanghai has many quite prosperous districts including Jingan District and Xuhui District, rent of coworking spaces we chose may beyond budget. Also, if this start-up company is a technical one, it may prefer a location with technology firms around. Therefore, the recommended zones should be taken as a starting point for further detailed analysis.

# ## Conclusion<a name="conclusion"></a>

# As a result, the 15 initial recommended locations can be find south-west from city center of Shanghai, most in Jingan District, based on constraints of number of coworking spaces, restaurants nearby and distance to nearest metro stations. 
# 
# While the recommended locations are just at the starting point, more detailed analysis and filtering should be done to find the optimal locations for start-up companies.

# 
