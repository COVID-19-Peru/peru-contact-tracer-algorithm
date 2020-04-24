__author__ = "Hugo Alatrista, Joseph Chamorro, Kristian Lopez, Miguel Nunez-del-Prado, Gonzalo Panizo"
__copyright__ = "Copyright 2020, The COVID-19 Tracer"
__credits__ = ["Hugo Alatrista", "Joseph Chamorro", "Kristian Lopez", "Miguel Nunez-del-Prado", "Gonzalo Panizo"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Joseph Chamorro"
__email__ = "jj.chamorrog@up.edu.pe"
__status__ = "Prof of Concept"

from google.cloud import bigquery as bq
import os
import sys
import configparser
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt
import pytz
import datetime
import math
import time
from sklearn.neighbors import BallTree
from math import sin, cos, sqrt, atan2, radians

global DISTANCE
global INTERVAL
global M
global T_START
global T_END
global OUTPUTPATH
global CREDENTIAL

def connectionDB():
    """ Get connection to the database"""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIAL
    client = bq.Client()
    return client

def getInfected():
    """ This method gets the infected list from the database"""
    client=connectionDB()
    query='''
          SELECT device_id FROM `cadi360-sac.kovid_dev.records` as records INNER JOIN `cadi360-sac.kovid_dev.infected` as supuestos
          USING(user_id);
          '''
    query_job = client.query(query)
    results = query_job.result()
    df = results.to_dataframe()
    infected=list(set(df['device_id'].to_list()))
    return infected

def getGPSrecords(t_start,t_end):
    """This method gets gps records of users from the database
       Input:
       ------
          @t_start : string
             start time
          @t_end : string
             end time
       Output:
       -------
          @GPSrecords : Dataframe
             GPS records of users that are between t_start and t_end from the database
    """
    client=connectionDB()
    query = '''
            select device_id as user,x as lon,y as lat,datetime
            from cadi360-sac.kovid_dev.records
            where datetime>=@t_start and datetime<=@t_end and x!=y and x is not null;
    '''
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ScalarQueryParameter('t_start', "STRING", t_start),
            bq.ScalarQueryParameter('t_end', "STRING", t_end),
        ]
    )
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()
    GPSrecords = results.to_dataframe()
    GPSrecords = gpd.GeoDataFrame(GPSrecords, geometry=gpd.points_from_xy(GPSrecords.lon, GPSrecords.lat))
    return GPSrecords

def getTrazaTimestamp(t_start,t_end,dataset,list_infected):
    """ This method gets GPS records of infected and uninfected users that are between t_start and t_end.
        Input:
        ------
           @t_start : string
              start time
           @t_end : string
              end time
           @dataset : Dataframe
              GPS records of users
           @list_infected : list
              list of infected users
        Output:
        -------
           @traza_uninfected : Dataframe
              GPS records of uninfected users that are between t_start and t_end
           @traza_infected : Dataframe
              GPS records of infected users that are between t_start and t_end
    """
    traza_uninfected = dataset[(dataset['datetime']>=t_start) & (dataset['datetime']<t_end) & (~dataset['user'].isin(list_infected))].reset_index(drop=True)
    traza_infected = dataset[(dataset['datetime']>=t_start) & (dataset['datetime']<t_end) & (dataset['user'].isin(list_infected))].reset_index(drop=True)
    return traza_infected,traza_uninfected

def getIntervals(t_start,t_end,interval):
    """ This method gets the list of dates between t_start and t_end with an interval in minutes
        Input:
        ------
           @t_start : string
              start time
           @t_end : string
              end time
           @interval : int
              interval between dates
        Output:
        -------
           @list_dates : list
              list of dates between t_start and t_end with an interval
    """
    timestamp_start = pd.Timestamp(t_start)
    timestamp_end = pd.Timestamp(t_end)
    nro_elements=int(int((timestamp_end-timestamp_start).total_seconds())/(60*interval))
    list_dates = []
    list_dates.append(t_start)
    for i in range(nro_elements):
        date_time_str = list_dates[i]
        Current_Date = dt.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        Next_Date = Current_Date + datetime.timedelta(minutes=interval)
        list_dates.append(str(Next_Date))
    return list_dates

def get_nearest(infected_coordinates, uninfected_coordinates, d):
    """ This method returns the indices and distances of the uninfected users that are within a distance "d"(paramater) of the infected users.
        Input:
        ------
           @infected_coordinates: array
              Latitude and lontitude of GPS coordinates of infected users.
           @uninfected_coordinates: array
              Latitude and lontitude of GPS coordinates of uninfected users.
           @d : int
              distance parameter
        Output:
        -------
           @indices : array
              indices of the uninfected users that are within a distance "d" of the infected users.
           @distances : array
              distance fron uninfected users to infected users.
    """
    # Create tree from the GPS coordinates of uninfected users
    tree = BallTree(uninfected_coordinates, leaf_size=15, metric='haversine')
    indices,distances=tree.query_radius(infected_coordinates, r=d,return_distance=True)
    indices=indices.transpose()
    distances=distances.transpose()
    return indices,distances


def nearest_neighbor(left_gdf, right_gdf, distance):
    """ This method returns the GPS events of infected users with uninfected users based on the distance "d"
        Input:
        ------
          @left_gdf : dataset
             Gps records of infected users
          @right_gdf : dataset =
             Gps records of unininfected users
          @distance : int
             distance parameter
        Output:
        -------
          @GPSevents : dataset
             GPS events of infected users with uninfected users based on the distance "d"
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    indices,meters = get_nearest(infected_coordinates=left_radians, uninfected_coordinates=right_radians , d=distance)
    infected=[];uninfected=[]
    for i in range(len(left_gdf)):
        list_indices=[];list_distances=[]
        list_indices = [i] * len(indices[i])
        points_infected = left_gdf.loc[list_indices]
        list_distances=(meters[i]*6371000).tolist()
        points_infected = points_infected.reset_index(drop=True)
        infected.append(points_infected.to_numpy())
        points_uninfected = right_gdf.loc[indices[i]]
        points_uninfected = points_uninfected.reset_index(drop=True)
        points_uninfected['distance']=list_distances
        uninfected.append(points_uninfected.to_numpy())
    array_infected = np.concatenate(infected,axis=0)
    array_uninfected = np.concatenate(uninfected,axis=0)
    events_infected = pd.DataFrame(array_infected,columns=["user_i","lon_i","lat_i","time_i","geometry"])
    events_uninfected= pd.DataFrame(array_uninfected,columns=["user_j","lon_j","lat_j","time_j","geometry","distance"])
    events_uninfected = events_uninfected.rename(columns={'geometry': 'closest_stop_geom'})
    GPSevents = events_infected.join(events_uninfected)

    return GPSevents

def filterEvents(intervals_dates,list_infected,distance):
    """ This method computes the GPS events of the infected users in each time interval
        based on a distance to return all the GPS events of the infected
        users (GPS coordinates of the infected users and their encounters with uninfected users)
        Input:
        ------
           @intervals_dates : list
              list of dates
           @list_infectad : list
              list of infected users
           @distance : int
              distance parameter
        Output:
        -------
           @GPSevents : dataframe
              GPS events of the infected users
    """
    d=distance
    list_gpsevents=[]
    for z in range(len(intervals_dates)-1):
        print("Interval: ",intervals_dates[z], "y", intervals_dates[z+1])
        infected,uninfected=getTrazaTimestamp(intervals_dates[z],intervals_dates[z+1],GPSrecords,list_infected)
        events_gps = nearest_neighbor(infected, uninfected, d)
        events_gps = events_gps.drop(['geometry','closest_stop_geom'], axis=1)
        print(len(events_gps))
        if(len(events_gps)!=0):
            list_gpsevents.append(events_gps.reset_index(drop=True))
        else:
            events_gps=pd.DataFrame()
            list_gpsevents.append(events_gps)
    #GPSevents=pd.concat(list_gpsevents).reset_index(drop=True)
    #return GPSevents
    return list_gpsevents

def probaContagius(lat1,lon1,lat2,lon2,M):
    """
        This method computes the distance lof latitudes and longitudes
        of two GPS coordinates in decimeters to return the contagius probability
        of the co-location represented by (lat1,lon1) and (lat2,lon2).
        Input:
        ------
           @lat1 : float
               Latitude of the first GPS coordinate (infected)
           @lat1 : float
               Longitude of the first GPS coordinate (infected)
           @lat2 : float
               Latitude of the second GPS coordinate
           @lat3 : float
               Longitude of the second GPS coordinate
           @M : numpy matrix
               Matrix containing the contagius probability based on distance
        Output:
        -------
           @proba : float
               Probabilty of being infected.
    """
    GAMMA = 0.02
    dlon = abs(lon2 - lon1) * 10000
    dlat = abs(lat2 - lat1) * 10000
    distance_Y = int(round(dlon, 0))
    distance_X = int(round(dlat, 0))
    proba = 0
    if ( (distance_X>=0 and distance_X<300) and (distance_Y>=0 and distance_Y<300) ):
        proba = GAMMA * M[distance_X][distance_Y]
    return proba

def componeProbs(p,p_prime):
    """
        This method compose two independent probabilities
        Input:
        ------
           @p : float
               Last computed probability
           @p_prime : float
               New computed probability

        Output:
        -------
           @proba : float
               Probability composition of the new and last probabilities
    """
    return p + p_prime * (1-p)

def NewHighRiskUsersGPS(list_infected):
    """
        This method saves the gps events for each time interval and then saves a file where the users are and their probability of being infected.
        Input:
        ------
           @list_infected : list
               list of infected users
    """
    d=DISTANCE/6371000
    intervals_dates=getIntervals(T_START,T_END,INTERVAL)
    encounters=filterEvents(intervals_dates,list_infected,d)
    prob=dict()
    cont=1
    for h in range(len(encounters)):
        if(len(encuentros[h])!=0):
            map_infected=encounters[h].iloc[::,[0,2,1,3]]
            map_noinfected=encounters[h].iloc[:,[0,2,1,4,6,5,7,8]]
            map_infected=map_infected.groupby( [ "user_i"] ).size().to_frame(name = 'count').reset_index()
            map_infected['prob']=[1.0]*len(map_infected)
            map_infected['lat_i']=[None]*len(map_infected)
            map_infected['lon_i']=[None]*len(map_infected)
            map_infected['time_i']=[None]*len(map_infected)
            map_infected['user_infectado']=[None]*len(map_infected)
            map_infected['lat_infectado']=[None]*len(map_infected)
            map_infected['lon_infectado']=[None]*len(map_infected)
            map_infected = map_infected[["user_i", "lat_i", "lon_i","time_i","count","prob","user_infectado","lat_infectado","lon_infectado"]]
            map_infected.columns=["user","lat","lon","time","datos","prob","user_infectado","lat_infectado","lon_infectado"]
            users=map_noinfected['user_j'].unique()
            dict_probability=dict()
            probability_df=[]
            for u in users:
                try:
                    dict_probability[u]=[prob[u]]
                except:
                    dict_probability[u]=[0]
            user=[];probability=[]
            for e in range(len(encounters[h])):
                probcontagio=probaContagius(encounters[h]['lat_i'][e],encounters[h]['lon_i'][e],encounters[h]['lat_j'][e],encounters[h]['lon_j'][e],M)
                probability_df.append(probcontagio)
                dict_probability[encounters[h]['user_j'][e]].append(probcontagio)
            map_noinfected['prob']=probability_df
            for d in dict_probability:
                for i in range(len(dict_probability[d])-1):
                    if(i==0):
                        q=componeProbs(dict_probability[d][i],dict_probability[d][i+1])
                    else:
                        q=componeProbs(q,dict_probability[d][i+1])
                user.append(d)
                probability.append(q)
            list_df=[]
            for i in range(len(user)):
                df=map_noinfected[map_noinfected['user_j']==user[i]].reset_index(drop=True)
                df['prob']=probability[i]
                list_df.append(df)
                prob[user[i]]=probability[i]
            map_noinfected=pd.concat(list_df)
            map_noinfected = map_noinfected[["user_j", "lat_j", "lon_j","time_j","distance","prob","user_i","lat_i","lon_i"]]
            map_noinfected.columns=["user","lat","lon","time","datos","prob","user_infectado","lat_infectado","lon_infectado"]
            data=pd.concat([map_infected,map_noinfected])
            data.to_csv(OUTPUTPATH+"eventos"+str(cont)+".csv",index=False)
            cont=cont+1
        else:
            data=pd.DataFrame(columns=["user","lat","lon","time","datos","prob","user_infectado","lat_infectado","lon_infectado"])
            data.to_csv(OUTPUTPATH+"eventos"+str(cont)+".csv",index=False)
            cont=cont+1
    data_probability=pd.DataFrame(prob.items(), columns=['user_id', 'Probability'])
    data_probability.to_csv(OUTPUTPATH+"probability.csv",index=False)

if __name__ == "__main__":
    if len(sys.argv)<=1:
        print("ERROR: You need to specify the path of the config file")
    else:
        now = dt.datetime.now()
        cfgName = sys.argv[1]
        config = configparser.ConfigParser()
        config.read(cfgName)
        logging.info("Reading configuration")
        print (config.sections())
        T_START = (config.get('parameters','t_start')).strip("'")
        T_END = (config.get('parameters','t_end')).strip("'")
        DISTANCE = int(config.get('parameters','distance'))
        INTERVAL = int(config.get('parameters','interval'))
        CREDENTIAL = (config.get('path','credential')).strip("'")
        MATRIX = (config.get('path','matrix')).strip("'")
        OUTPUTPATH = (config.get('path','outputFilePath')).strip("'")
        M = np.load(MATRIX)
        GPSrecords = getGPSrecords(T_START,T_END)
        list_infected = getInfected()
        NewHighRiskUsersGPS(list_infected)
        then = dt.datetime.now()
        duration = then-now
        print(duration.total_seconds())
