#!/usr/bin/env python
# coding: utf-8
## Copyright:
## Gonzalo Panizo García
## gonzalo.panizo@gmail.com
## April 7, 2020

# In[2]:


import numpy as np
import pandas as pd
import datetime as dt
from numpy import random as rnd
from numba import jit, njit, prange
import scipy.interpolate
import glob
#import pytz


# In[2]:


#for i in range(1,3):
#    G_U   = pd.read_csv('AccesoBaseDatos/gps_20200507_' + str(i) + '.csv')
#    U_aux = G_U[G_U['type'] == 'REGISTER'][['datetime', 'device_id', 'user_id']]
#    U_    = pd.concat([U_, U_aux])
#    print(len(U_aux))


# In[3]:


Ti      = dt.datetime.strptime('21/4/20', '%d/%m/%y')                     # tiempo inicial del análisis
T       = dt.datetime.strptime('8/5/20', '%d/%m/%y')                      # tiempo final del análisis
#Ti      = dt.datetime(2020, 4, 21, tzinfo=pytz.timezone('America/Lima'))  # tiempo inicial del análisis
#T       = dt.datetime(2020, 5, 7, tzinfo=pytz.timezone('America/Lima'))   # tiempo final del análisis
T_segs  = (T - Ti).total_seconds()                                        # en segundos
t_enf   = 14                                                              # duracion de la enfermedad en dias
p_cont  = 0.12                                                            # probab de recibir dosis de contagio   
dc      = 15                                                              # distancia de contagio
sigma   = 35                                                              # error medio de la medicion GPS
delta_x = 300                                                             # distancia de vecindad en decimetros
delta_y = delta_x                                                         # distancia de vecindad en decimetros
delta_t = 200                                                             # distancia de vecindad temporal en segs
fxyt    = delta_x/np.sqrt(delta_t/2)                                      # factor de difusión espacio-tiempo (MB)
n       = 10000000                                                        # numero de simulaciones para gausiana
nt      = delta_t                                                         # numero de tiempos a simular
nx      = int(delta_x/10)                                                 # numero de posiciones a simular (c/10)
x_lima, y_lima = -77.042793, -12.046374                                   # centro de Lima en long-lat (grados)

# 'calific' = calificacion de usuario: no infectado = 0, infectado dudoso = 1, infectado seguro = 2, 
# recuperado = 3, con propensión a infectarse = -1, con riesgo adicional (mobilidad) = -2, etc.
# chequear que todo I y todo G esten en U, sino dar alerta.
# falta la fecha desde la que el paciente es contagioso (por ahora se elige un numero aleatorio entre 1 y 14)
# las fechas en I (fecha del test) y en G (posición GPS) tiene distinta localizacion


# In[4]:


def infectados_ini(I_):
    I1 = I_[~I_['status'].isna()].drop_duplicates('user_id')          # depura lineas con datos 'status' (=test)
    I2 = I_[~I_.user_id.isin(I1.user_id)].drop_duplicates('user_id')  # idem para el resto de líneas
    I3 = pd.concat([I1, I2])                                          # une ambas listas

    I3.loc[:, 'calific'] = [2 for i in I3.index]        # lista de infectados, por defecto infectado seguro = 2
    I3.loc[I3['status'] == 'RAPIDO',          'calific'] = 1          # no tan seguro
    I3.loc[I3['status'] == 'NO ESPECIFICADO', 'calific'] = 2          # por defecto seguro
    I3.loc[I3['status'] == 'HISOPADO',        'calific'] = 2
    I3.loc[I3['status'] == 'POSITIVO',        'calific'] = 2

    I3.loc[:, 'domicx_dm'] = ((I3['x'] - x_lima) * 1114128.4 * np.cos(I3['y']*np.pi/180))  # posic de
    I3.loc[:, 'domicy_dm'] = ((I3['y'] - y_lima) * 1111329.15)        # domicilio en dm (resp al cent de Lima)

    #I3.loc[:, 't_test']    = (I3['date'] - Ti.replace(tzinfo=None)).dt.total_seconds()  
    I3.loc[:, 't_test']    = (pd.to_datetime(I3['date'], format='%Y-%m-%d') - Ti).dt.total_seconds()  
                                                                      # tiempo del test en segs resp a Ti
    I3.loc[:, 't_infec']   =  I3['t_test'] - 24*3600*14               # t empieza a infectar 
    #I3.loc[:, 't_infec']   =  I3['t_test'] - 24*3600*np.random.randint(t_enf, size=len(I3))  
    I3.loc[:, 't_sano']    =  I3['t_infec'] + 24 * 3600 * t_enf       # tiempo en que ya no infecta

    I  = I3[I3.user_id.isin(U_.user_id)]                    # restringe a infectados en lista de usuarios
    return I


# In[5]:


def usuarios_ini(U_, I):
    U1 = U_.drop_duplicates(['device_id'])     # depura lista de usuarios por equipo porque en G solo hay equipos
    U  = U1.merge(I[['user_id', 'domicx_dm', 'domicy_dm', 't_infec', 't_sano', 'calific']], 
                  how='left', on='user_id')    # agrega datos de los infectados en U, en particular calific > 0
                                                  
    U.loc[U['calific'].isna(), 'calific'] = 0  # como no hay 'calific' previo se asume que el resto son sanos
    return U


# In[6]:


def gps_ini(G_, U):
    G1 = G_.drop_duplicates()                         # elimina registros duplicados
    G1 = G1.drop(G1[G1['type'] == 'REGISTER'].index)  # elimina registros REGISTER (usuario se registra)
    G1 = G1[G1.device_id.isin(U.device_id)]           # restringe a equipos en lista de usuarios
    G1 = G1.drop(G1[G1['x'] == G1['y']].index)        # depura un error comun en los datos GPS
    G1 = G1.reset_index().merge(U[['device_id', 'domicx_dm', 'domicy_dm', 't_infec', 't_sano', 'calific']], 
               how='left', on='device_id').set_index('index')  # agrega datos en G para el calculo de probabilidad
    G1['x_dm']   = (G1['x'] - x_lima) * 1114128.4 * np.cos(G1['y']*np.pi/180)      # posicion en dm (resp 
    G1['y_dm']   = (G1['y'] - y_lima) * 1111329.15                                 # a centro de Lima)
    #G1['t_segs'] = (G1['datetime'] - Ti.replace(tzinfo=None)).dt.total_seconds()  # tiempo en segundos
    G1['t_segs'] = (pd.to_datetime(G1['datetime'], format='%Y-%m-%d %H:%M:%S') - Ti).dt.total_seconds()  
                                                                                   # tiempo en segundos

    G1 = G1[(0 <= G1['t_segs']) & (G1['t_segs'] <= T_segs)]   # solo datos en intervalo de análisis
    G  = G1.drop(G1[((G1['t_infec'] > G1['t_segs']) | (G1['t_segs'] > G1['t_sano'])) & 
                          (G1['calific'] >= 1)].index)  # elimina posiciones GPS de infectados cuando no contagian 
    return G


# In[7]:


def calcula_cercanias(G, file):
    #min_x   = np.min(G['x_dm'])
    #min_y   = np.min(G['y_dm'])
    G['xr'] = G['x_dm'] // delta_x            # partición de coordenadas x en intervalos de longitud delta_x
    G['yr'] = G['y_dm'] // delta_y            # partición de coordenadas y en intervalos de longitud delta_y
    G['tr'] = G['t_segs'] // delta_t          # partición de coordenadas t en intervalos de longitud delta_t

    Ge   = G[G['calific'] >= 1]               # grupo de enfermos
    Gs   = G[G['calific'] <= 0]               # grupo de sanos

    Gs   = Gs.reset_index()                   # guarda las posiciones en la base de datos
    Ge   = Ge.reset_index()                   # guarda las posiciones en la base de datos

    u    = [-1,0,1]                           # posibles direcciones de intervalos vecinos de las particiones
    lv   = [[i, j, k] for i in u for j in u for k in u if not i == j == k == 0]  # lista de direcciones en 3D
    lf   = [Ge]
    for v in lv:                        # agrega particiones vecinas 3D a Ge (paralelepipedos cerca a infectados)
        Gec = Ge.copy()                 # copia independiente que se puede modificar sin alterar el original
        Gec.loc[:, ['xr','yr','tr']] = np.array(Ge[['xr','yr','tr']] + v)
        lf.append(Gec)
    
    Gc  = pd.concat(lf)
    Gss = Gs[['index', 'device_id', 'domicx_dm', 'domicy_dm', 'calific', 'x_dm', 'y_dm', 't_segs', 
              'xr', 'yr', 'tr']]
    Gee = Gc[['index', 'device_id', 'domicx_dm', 'domicy_dm', 'calific', 'x_dm', 'y_dm', 't_segs', 
              'xr', 'yr', 'tr']]

    Eg = Gss.merge(Gee, how='inner', on=['xr', 'yr', 'tr'])  # contactos (eventos de cercania sano-enfermo)
    Eg.loc[:, 'file'] = file
    return Eg


# In[76]:


@njit(parallel = True)
def prop_cerca(i, a, y1, x2, y2):
    x1 = i + sigma * a
    count = np.sum((x1 - x2)**2 + (y1 - y2)**2 < dc**2)
    return count/n

a = rnd.randn(n)
b = rnd.randn(n)
c = rnd.randn(n)
d = rnd.randn(n)
e = rnd.randn(n)
f = rnd.randn(n)

def distrib_gauss(nt, nx):
    mat  = np.zeros((nt,nx))
    y1   = sigma * b
    for t in prange(nt):
        x2 = sigma * c + fxyt * np.sqrt(t) * e
        y2 = sigma * d + fxyt * np.sqrt(t) * f
        for i in prange(nx):
            mat[t, i] = prop_cerca(10*i, a, y1, x2, y2)       
    return mat

D = distrib_gauss(nt, nx + 1)  # uno mas en dimensión espacial para poder interpolar en todo el rango


# In[81]:


def proba(t, x):
    vx = [10*i for i in range(D.shape[1])]
    if t >= D.shape[0] or x > 10*(D.shape[1]-1):
        return 0
    else:
        vy = D[int(t),:]
        y_interp = scipy.interpolate.interp1d(vx, vy)
    return float(y_interp(x))


# In[108]:


I_ = pd.read_csv('AccesoBaseDatos/infected_20200516.csv')
U_ = pd.read_csv('AccesoBaseDatos/usuarios_20200507.csv')
I = infectados_ini(I_)
U = usuarios_ini(U_, I)

files_gps = glob.glob("AccesoBaseDatos/gps_20200*.csv")
lista = []
for file in files_gps:
    G_ = pd.read_csv(file)
    G  = gps_ini(G_, U)
    Gc = calcula_cercanias(G, file)
    lista.append(Gc)
    print(file)


# In[109]:


Eg1 = pd.concat(lista)


# In[110]:


Eg2 = Eg1.drop_duplicates(['device_id_x', 'x_dm_x', 'y_dm_x', 't_segs_x', 
                           'device_id_y', 'x_dm_y', 'y_dm_y', 't_segs_y'])


# In[53]:


#Eg2['dist'] = np.sqrt((Eg2['x_dm_x'] - Eg2['x_dm_y'])**2 + (Eg2['y_dm_x'] - Eg2['y_dm_y'])**2)
#Eg2['t_dist'] = np.abs(Eg2['t_segs_x'] - Eg2['t_segs_y'])


# In[111]:


#Eg2      = Eg2.drop(Eg2[(Eg2['x_dm_x'] - Eg2['x_dm_y'])**2 + (Eg2['y_dm_x'] - Eg2['y_dm_y'])**2 >= 300**2].index)
# esto de arriba funciona mal, averiguar causa (no es importante).
Eg          = Eg2  # falta medir la longitud del intervalo de tiempo de contacto para un mejor calculo de la proba
Eg['proba'] = Eg[['t_segs_x', 't_segs_y', 'x_dm_x', 'x_dm_y', 'y_dm_x', 'y_dm_y']].apply(lambda x:
            p_cont * proba(np.abs(x[0] - x[1]).astype(int), np.sqrt((x[2] - x[3])**2 + (x[4] - x[5])**2)), axis=1)


# In[113]:


#Eg[Eg['device_id_x'] == 'e960dcd2-ab7d-451c-aaed-8ea6cad2f339'][[
#    't_segs_x', 't_segs_y', 'x_dm_x', 'x_dm_y', 'y_dm_x', 'y_dm_y', 'dist', 't_dist', 'proba']]


# In[114]:


P = Eg.groupby('device_id_x').agg({'proba' : lambda x: 1 - np.prod(1-x)})


# In[123]:


print(P[P['proba'] != 0].to_string())


# In[ ]:




