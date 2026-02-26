# %%
import sys
import json

general = {'step_size': '0.001', 'use_road_network': True, 'coef_day_to_year': '300', 'clustering_radius': '500'}

params = {
    'general': general,
    }

default = {'training_folder': '../../scenarios/guide_utilisateur_lyon', 'params':params} # Default execution parameters
manual, argv = (True, default) if 'ipykernel' in sys.argv[0] else (False, dict(default, **json.loads(sys.argv[1])))
print(argv)

## TODO: renvoyer vers la doc dans le paragraphe de description des paramètres

# %%
import os
import time
import geopandas as gpd
import pandas as pd
pd.set_option('display.max_columns', 50)
sys.path.insert(0, r'../../../quetzal') # Add path to quetzal
import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from typing import Literal
import numba as nb
from collections import defaultdict
from sklearn.cluster import DBSCAN
import shapely
#num_cores = 1
print('numba threads',nb.config.NUMBA_NUM_THREADS)

on_lambda = bool(os.environ.get('AWS_EXECUTION_ENV'))
io_engine = 'pyogrio' 

# %%
sys.path.insert(0, r'../../') # Add path
from utils import get_epsg, population_to_mesh, get_acf_distances, get_routing_distances

# %% [markdown]
# # Folders stucture and params

# %% [markdown]
# Everything is on S3 (nothing on ECR) so no direct input folder. just scenarios/{scen}/inputs/

# %%
argv['training_folder']

# %%
argv['params']

# %%
base_folder = argv['training_folder']
input_folder = os.path.join(base_folder,'inputs/')
pt_folder  = os.path.join(input_folder,'pt/')
road_folder = os.path.join(input_folder,'road/')
od_folder =  os.path.join(input_folder,'od/')

output_folder = os.path.join(base_folder,'outputs/')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model_folder = os.path.join(input_folder, 'model/')

# %%
# Read general params
step_size_min = 0.0005
step_size = max(float(argv['params']['general'].get('step_size')), step_size_min)
use_road_network = argv['params']['general'].get('use_road_network')
coef_day_to_year = float(argv['params']['general'].get('coef_day_to_year'))
clustering_radius = float(argv['params']['general'].get('clustering_radius'))

# %%
# Default pt_links params in case not filled by the user
default_catchment_radius = 500      # meters
default_capex = 0.3                 # €/veh.km
default_capacity = 60               # vehicle capacity in PAX
default_service_hours = 12          # nb d'heures par jour de fonctionnement
default_road_pt_factor = 1.25       # facteur de dilatation des vitesses routières pour les TC utilisant la route (bus, bus express...)
default_stop_interval = 300         # temps d'arrêt à chaque arrêt
#TODO: replace stop_interval with dwell_time ?
default_modal_share = 0.2           # modal share for the mode (used only if ODs are provided)
#TODO: voir si on peut ajouter les champs défaut dans les liens ==> suppr. ==> adapter script
#TODO: changer capex en opex à travers le script parce que ça n'a pas de sens.
#TODO: plutôt que de mettre des valeurs par défaut qui induisent en erreur, mieux vaudrait identifier quelles valeurs sont essentielles à la 
# capture de tel ou tel indicateur (quand c'est possible) et n'afficher ces indicateurs que si les bons intrants sont renseignés.
#TODO: reconsidérer certaines des valeurs par défaut (par exemple les valeurs liées à la dilatation du temps, ne vaudrait-il pas mieux considérer par défaut qu'il n'y en a pas ? ie stop time = 0 et dilatation =1)

# %% [markdown]
# # Inputs

# %% [markdown]
# PT links

# %%
### TODO: ajouter test que les trip_ids sont bien renseignés correctement:
### TODO: ajouter correctifs sur tous les champs qui doivent être des nombres

with open(pt_folder + 'links.geojson') as f:
    links_ = json.load(f)

columns = set()
for feature in links_['features']:
    for key in feature['properties'].keys():
        columns.add(key)

links = pd.DataFrame(links_['features'])
for col in columns:
    links[col] = links.apply(lambda x: x['properties'].get(col, None), 1)
links['geometry'] = links['geometry'].apply(lambda x: LineString(x['coordinates']))
links.drop(columns=['type', 'properties'], inplace=True)

links = links.set_index('index')
links = gpd.GeoDataFrame(links, geometry='geometry', crs='EPSG:4326')

# %%
## TODO: insert integrity tests

# %%
nodes = gpd.read_file(pt_folder + 'nodes.geojson', engine=io_engine)
nodes = nodes.set_index('index')
nodes = nodes[~pd.isna(nodes.geometry)]

nodes['nindex'] = nodes.reset_index().index
nodes['stop_name'] = nodes.apply(lambda x: x['nindex'] if (pd.isna(x['stop_name']) or x['stop_name'] is None) else x['stop_name'], 1).astype(str)
nodes.drop(columns='nindex', inplace=True)

# %%
## TODO: insert node integrity tests

# %%
prep_nodes_to_route_id_a = links.copy()[['a', 'trip_id', 'route_id']]
prep_nodes_to_route_id_b = links.copy()[['b', 'trip_id', 'route_id']]
prep_nodes_to_route_id_a.rename(columns={'a': 'node'}, inplace=True)
prep_nodes_to_route_id_b.rename(columns={'b': 'node'}, inplace=True)

prep_nodes_to_route_id = pd.concat([prep_nodes_to_route_id_a, prep_nodes_to_route_id_b])

# %%
nodes_to_trip_ids = prep_nodes_to_route_id.groupby('node')['trip_id'].agg(set).to_dict()
nodes_to_route_ids = prep_nodes_to_route_id.groupby('node')['route_id'].agg(set).to_dict()

# %%
nodes['trip_ids'] = nodes.index.map(nodes_to_trip_ids)
nodes['route_ids'] = nodes.index.map(nodes_to_route_ids)

# %%
## TODO: cette cellule est bizarre, on dirait que tout n'a pas été fait correctement, pourquoi on prend forcément des valeurs de base pout la pointe, alors
# qu'il faudrait que par défaut on ait partout headway_ph = headway_oph = headway...

## TODO: gérer la mise à jour du champ headway avec une valeur par défaut en cas d'absence (ou faire apparaître un message d'erreur et interrompre si 
# chaque ligne n'a pas un headway)

## TODO: make sure headway is consistent (one headway for both directions)

links['headway'] = links['headway'].astype(float)
## TODO: ajouter les valeurs par défaut ->
# - pour chaque trip minimum sur le min ou max ?? sur le route_id
# - si pas de valeur sur le route id, chercher sur tout le réseau et renvoyer un warning
# - si pas de valeur sur tout le réseau, valeur par défaut 600 et warning

default = {'headway_ph': links['headway'].astype(float).max(),
           'headway_oph': links['headway'].astype(float).max(),
           'nb_service_hours': 14,
           'nb_peak_hours': 14}

## TODO: to be adapted: why are service hours 14 and peak hours 14 ? can i set a certain number of service hours and no peak hours ? 
# I.e. it would be good if default nb_peak_hours was set based on nb_service hours

## TODO: show a warning whenever a default value is set
## TODO: for all columns, fix potential NaNs and shoot a warning

for col in ['headway_ph', 'headway_oph']:
    print(col)
    if col not in links.columns:
        links[col] = default[col]
    links.loc[links[col].isna(), col] = links.loc[links[col].isna(), 'headway']
    links[col] = links[col].astype(float)

for col in ['nb_service_hours', 'nb_peak_hours']:
    if col not in links.columns:
        links[col] = default[col]
    else:
        links[col] = links[col]
        dic_col = links.groupby('route_id')[col].max()
        links[col] = links['route_id'].map(dic_col).fillna(default[col])
        links[col] = links[col].astype(int)

# %%
if 'capacity' not in links.columns:
    links['capacity'] = default_capacity
else:
    ## TODO: replicate for all default inputs.
    undefined = links.loc[links['capacity'].isna(), 'trip_id'].unique
    links['capacity'] = links['capacity'].fillna(default_capacity).astype(float)
    print('capacity was not defined for the following lines: {}'.format(list(undefined)))
    print('Default value used instead: {}'.format(default_capacity))

# %%
## TODO: see capacity
if 'capex' not in links.columns:
    links['capex'] = default_capex
else:
    links['capex'] = links['capex'].fillna(default_capex).astype(float)

# %%
catchment_radii = [x for x in links.columns if 'catchment_radius' in x]
catchment_radii_provided = (len(catchment_radii) > 0)
if catchment_radii_provided:
    for x in catchment_radii:
        links[x] = links[x].astype(float)
        M = links[x].max()
        if not links[x].equals(links[x].fillna(M)):
            print('!! Catchemnt radius values missing in column {} !!'.format(x))
            ##TODO: add default value
            links[x] = links[x].fillna(M).astype(float)
else:
    links['catchment_radius'] = default_catchment_radius

# %%
## TODO: see capacity
if 'nb_service_hours' not in links.columns:
    links['nb_service_hours'] = default_service_hours
else:
    links['nb_service_hours'] = links['nb_service_hours'].fillna(default_service_hours).astype(float)

# %%
## TODO: see capacity
if 'departures' not in links.columns:
    links['departures'] = None
if 'arrivals' not in links.columns:
    links['arrivals'] = None

# %%
## TODO: see capacity
if 'road_pt_factor' not in links.columns:
    links['road_pt_factor'] = default_road_pt_factor
links['road_pt_factor'] = links['road_pt_factor'].fillna(default_road_pt_factor).astype(float)

if 'stop_interval' not in links.columns:
    links['stop_interval'] = default_stop_interval
links['stop_interval'] = links['stop_interval'].fillna(default_stop_interval).astype(float)

# %% [markdown]
# Input data zoning file

# %%
# find meters CRS
centroid = [*LineString(nodes.centroid.values).centroid.coords][0]
crs = get_epsg(centroid[1],centroid[0])
crs

# %%
zonage_file = os.path.join(input_folder, 'zonage.geojson')
zonage_file_provided = os.path.isfile(zonage_file)
if zonage_file_provided :
    zonage_ = gpd.read_file(input_folder + 'zonage.geojson', engine=io_engine).to_crs(epsg='4326')
    zonage_['area (km2)'] = zonage_.to_crs(crs).area / 10**6
    if 'zone_id' not in zonage_.columns:
        zonage_.reset_index(names='zone_id', inplace=True)
else:
    print('No zonage file in the input folder...')

## TODO: il faut se demander ce qu'il serait possible de faire sans zonage, voire comment gérérer le zonage directement sans 
# avoir besoin d'en fournir (SYSMAP)

# %%
##TODO: only if zones are provided
nodes_with_zones = nodes.copy().sjoin(zonage_, predicate='intersects', how='left')[['stop_name', 'route_ids', 'trip_ids', 'zone_id']]
nodes_with_zones['route_ids'] = nodes_with_zones['route_ids'].apply(lambda x: list(x))
nodes_with_zones['trip_ids'] = nodes_with_zones['trip_ids'].apply(lambda x: list(x))
nodes_with_zones.reset_index(inplace=True)
nodes_per_zone = nodes_with_zones.groupby('zone_id')[['stop_name', 'route_ids', 'trip_ids', 'index']].agg({'stop_name': set, 'route_ids': sum, 'trip_ids': sum, 'index': set}).to_dict()

zonage = zonage_.copy()
zonage['route_ids'] = zonage['zone_id'].map(nodes_per_zone['route_ids']).apply(lambda x: (x if isinstance(x, set) else set(x)) if not isinstance(x, float) else set())
zonage['trip_ids'] = zonage['zone_id'].map(nodes_per_zone['trip_ids']).apply(lambda x: (x if isinstance(x, set) else set(x)) if not isinstance(x, float) else set())
zonage['stop_names'] = zonage['zone_id'].map(nodes_per_zone['stop_name']).apply(lambda x: (x if isinstance(x, set) else set(x)) if not isinstance(x, float) else set())
zonage['stop_ids'] = zonage['zone_id'].map(nodes_per_zone['index']).apply(lambda x: (x if isinstance(x, set) else set(x)) if not isinstance(x, float) else set())

# %%
densities = [x for x in zonage.columns if 'density' in x]
assert len(densities) > 0, 'Please provide densities as input data in the zoning file'

# %%
zonage.to_crs(epsg=4326).to_file(output_folder + 'zoning.geojson')

# %%
display_ph_columns = ('headway_ph' in links.columns)
## If not display_ph_columns, no peak hour considered

# %% [markdown]
# Road network

# %%
## road network here
rnodes_file = os.path.join(road_folder, 'road_nodes.geojson')
rnodes_file_provided = os.path.isfile(rnodes_file)
if rnodes_file_provided:
    rnodes = gpd.read_file(os.path.join(road_folder, 'road_nodes.geojson'), engine=io_engine)
    rnodes = rnodes.set_index('index').to_crs(epsg='4326')
    rlinks = gpd.read_file(os.path.join(road_folder, 'road_links.geojson'), engine=io_engine)
    rlinks = rlinks.set_index('index').to_crs(epsg='4326')
print('road network ?',rnodes_file_provided)

# %%
## TODO: insert road integrity tests

# %% [markdown]
# OD

# %%
## TODO: OD file tests ?

od_file = os.path.join(od_folder, 'od.geojson')
od_file_provided = os.path.isfile(od_file)
if od_file_provided:
    od = gpd.read_file(od_file, engine=io_engine)
    if 'index' not in od.columns:
        od.reset_index(names='index', inplace=True)
    if 'name' not in od.columns:
        od['name'] = od['index'].astype(str)
    od['name'] = od['name'].fillna(od['index'].astype(str))

    od.drop_duplicates(inplace=True)
else:
    print('OD?', od_file_provided)

if od_file_provided:
    if 'modal_share_input' not in links.columns:
        links['modal_share_input'] = default_modal_share
    links['modal_share_input'] = links['modal_share_input'].fillna(default_modal_share).astype(float)
    

# %% [markdown]
# # Init result dataframes

# %%
## Init route results (several directions)
df_route_id = pd.DataFrame(index=links['route_id'].unique())
df_route_id.index.name = 'route_id'
df_route_id = df_route_id.reset_index()

if display_ph_columns:   
    df_route_id = df_route_id.merge(links[['route_id', 'route_type', 'capacity', 'headway', 'headway_ph', 'headway_oph', 'nb_peak_hours']], on='route_id', how='left')
else:
    df_route_id = df_route_id.merge(links[['route_id', 'route_type', 'capacity', 'headway']], on='route_id', how='left')

df_route_id = df_route_id.rename(columns={'capacity': 'veh_capacity (PAX)'})
df_route_id = df_route_id.drop_duplicates()
df_route_id = df_route_id.set_index('route_id')

# %%
## Init trip results (one direction)
## TODO: make sure trip_ids are consistent
df_trip_id = pd.DataFrame(index=links['trip_id'].unique())
df_trip_id.index.name = 'trip_id'
df_trip_id = df_trip_id.reset_index()

if display_ph_columns:   
    df_trip_id = df_trip_id.merge(links[['trip_id', 'route_id', 'route_type', 'capacity', 'headway', 'headway_ph', 'headway_oph', 'nb_peak_hours']], on='trip_id', how='left')
else:
    df_trip_id = df_trip_id.merge(links[['trip_id', 'route_id', 'route_type', 'capacity', 'headway']], on='trip_id', how='left')

df_trip_id = df_trip_id.rename(columns={'capacity': 'veh_capacity (PAX)'})
df_trip_id = df_trip_id.drop_duplicates()
df_trip_id = df_trip_id.set_index('trip_id')

# %%
# Init route type results (all routes with same route type; eg bus, train...)
df_route_type = pd.DataFrame(index=links['route_type'].unique())
df_route_type.index.name='route_type'

# %%
# Make sure headways are consistent : one single headway for both way and return
## TODO: fix in links
# Otherwise can't calculate KPIs later

df_route_id = df_route_id[~df_route_id.index.duplicated(keep='first')]
route_headway = dict(zip(df_route_id.index, df_route_id['headway']))
links.headway = links.route_id.map(route_headway)

# %% [markdown]
# # Geometries

# %%
# Trip geometries
geoms_trip_id = links.groupby('trip_id')['geometry'].agg(shapely.unary_union).to_dict()
df_trip_id['geometry'] = geoms_trip_id

# %%
# Route geometries (direction = 0)
geoms_route_id = links.copy()
geoms_route_id['trip_number'] = geoms_route_id['trip_id'].apply(lambda x: x[-1])
geoms_route_id = geoms_route_id.loc[geoms_route_id['trip_number'] =='0']
geoms_route_id = geoms_route_id.groupby('route_id')['geometry'].agg(shapely.unary_union).to_dict()
df_route_id['geometry'] = geoms_route_id

# %% [markdown]
# # Catchment calculation

# %%
def get_catchment_by_mode(col='route_id', pop_col='population', node_dist=None):
    #get all nodes with col filter
    link = links.groupby(col)[['a','b','route_type', 'catchment_radius']].agg({'a':set,'b':set,'route_type':'first', 'catchment_radius': 'first'})
    link['node'] = link.apply(lambda row: row['a'].union(row['b']), axis=1)
    link = link.drop(columns=['a','b'])

    col_exist = col == 'route_type' # cannot explode if index == route_type (a column)
    link = link.explode('node').reset_index(drop=col_exist)
    link = node_dist.merge(link, left_on='node_index', right_on='node')
    #filter by distance
    link = link[link['distances'] <= link['catchment_radius']]
    #drop duplicated mesh nodes (we count only one time)
    link = link.drop_duplicates(subset=['mesh_index',col],keep='first')

    return link.groupby(col)[pop_col].sum().to_dict()

# %%
def get_catchment_by_access(col='route_id', pop_col='population', catchment_col='catchment_radius', node_dist=None):
    #get all nodes with col filter
    link = links.groupby(col)[['a','b','route_type', catchment_col]].agg({'a':set,'b':set,'route_type':'first', catchment_col:'first'})
    link['node'] = link.apply(lambda row: row['a'].union(row['b']), axis=1)
    link = link.drop(columns=['a','b'])

    col_exist = col == 'route_type' # cannot explode if index == route_type (a column)
    link = link.explode('node').reset_index(drop=col_exist)
    link = node_dist.merge(link, left_on='node_index', right_on='node')
    #filter by distance
    # link[catchment_col] = link[catchment_col].astype(float)
    link = link[link['distances'] <= link[catchment_col]]
    #drop duplicated mesh nodes (we count only one time)
    link = link.drop_duplicates(subset=['mesh_index',col],keep='first')

    return link.groupby(col)[pop_col].sum().to_dict()

# %%
rnodes.crs

# %%
## TODO: comment le catchment gère-t-il les histoires de recouvrement entre différents noeuds pour les catchment radius?
# peut-être pas pertinent pour le catchment ligne par ligne de faire le catchment point par point mais générer un buffer commun, sinon
# on va avoir des redondances.

## TODO: vérifier comment les correspondances sont gérées en l'absence de timetable, en particulier
# comment l'heure de pointe est gérée pour le calcul des temps d'attente en correspondance.

## TODO: les valeurs sont étrangement faibles en utilisant le réseau routier...

meshes = {}
node_dists = {}

## road network here
for density in densities:
    if density == 'density' and 'population_density' not in densities:
        tag = 'population'
    elif density == 'density':
        tag = 'x'
    else:
        tag = density.split('_density')[0]

    print(tag)

    zonage[tag] = zonage[density] * zonage['area (km2)']

    if rnodes_file_provided and use_road_network:
        # use rnodes as mesh_pop.
        print('using road_nodes') ## road network here
        mesh = population_to_mesh(zonage, mesh=rnodes, step=step_size, col=tag, fill_missing='closest')
    else:
        # create a mesh
        mesh = population_to_mesh(zonage, mesh=None, step=step_size, col=tag, fill_missing='centroid')

    #mesh.to_file(output_folder + 'population_mesh.geojson',driver='GeoJSON',engine=io_engine)
    if catchment_radii_provided:
        max_dist = (max([links[catchment_radius].max() for catchment_radius in catchment_radii]))
    else:
        max_dist = default_catchment_radius

    meshes[tag] = mesh.copy()

    # road network here, what to use if not road network?
    if rnodes_file_provided: 
        print('using road_nodes')
        node_dist = get_routing_distances(nodes, rnodes, rlinks, mesh, tag, 'length', max_dist)
    else:
        node_dist = get_acf_distances(nodes, mesh, tag, crs, max_dist)

    node_dists[tag] = node_dist.copy()
    
    if catchment_radii_provided:
        for catchment_radius in catchment_radii:
            suf = catchment_radius.split('catchment_radius_')[1]

            res_trip = get_catchment_by_access('trip_id', tag, catchment_radius, node_dist)
            res_route = get_catchment_by_access('route_id', tag, catchment_radius, node_dist)
            res_mode = get_catchment_by_access('route_type', tag, catchment_radius, node_dist)

            if suf == '':
                df_trip_id['catchment {}'.format(tag)] = res_trip
                df_trip_id['catchment {}'.format(tag)] = df_trip_id['catchment {}'.format(tag)].fillna(0)
                df_trip_id['catchment {}'.format(tag)] = df_trip_id['catchment {}'.format(tag)].apply(lambda x: round(x, 0)) 

                df_route_id['catchment {}'.format(tag)] = res_route
                df_route_id['catchment {}'.format(tag)] = df_route_id['catchment {}'.format(tag)].fillna(0)
                df_route_id['catchment {}'.format(tag)] = df_route_id['catchment {}'.format(tag)].apply(lambda x: round(x, 0)) 

                df_route_type['catchment {}'.format(tag)] = res_mode
                df_route_type['catchment {}'.format(tag)] = df_route_type['catchment {}'.format(tag)].fillna(0)
                df_route_type['catchment {}'.format(tag)] = df_route_type['catchment {}'.format(tag)].apply(lambda x: round(x, 0)) 

            else:
                df_trip_id['catchment {} {}'.format(tag, suf)] = res_trip
                df_trip_id['catchment {} {}'.format(tag, suf)] = df_trip_id['catchment {} {}'.format(tag, suf)].fillna(0)
                df_trip_id['catchment {} {}'.format(tag, suf)] = df_trip_id['catchment {} {}'.format(tag, suf)].apply(lambda x: round(x, 0))

                df_route_id['catchment {} {}'.format(tag, suf)] = res_route
                df_route_id['catchment {} {}'.format(tag, suf)] = df_route_id['catchment {} {}'.format(tag, suf)].fillna(0)
                df_route_id['catchment {} {}'.format(tag, suf)] = df_route_id['catchment {} {}'.format(tag, suf)].apply(lambda x: round(x, 0))  

                df_route_type['catchment {} {}'.format(tag, suf)] = res_mode
                df_route_type['catchment {} {}'.format(tag, suf)] = df_route_type['catchment {} {}'.format(tag, suf)].fillna(0)
                df_route_type['catchment {} {}'.format(tag, suf)] = df_route_type['catchment {} {}'.format(tag, suf)].apply(lambda x: round(x, 0)) 

    else:
        res_trip = get_catchment_by_mode('trip_id', tag, node_dist)
        res_route = get_catchment_by_mode('route_id', tag, node_dist)
        res_mode = get_catchment_by_mode('route_type', tag, node_dist)

        df_trip_id['catchment {}'.format(tag)] = res_trip
        df_trip_id['catchment {}'.format(tag)] = df_trip_id['catchment {}'.format(tag)].fillna(0)
        df_trip_id['catchment {}'.format(tag)] = df_trip_id['catchment {}'.format(tag)].apply(lambda x: round(x, 0)) 

        df_route_id['catchment {}'.format(tag)] = res_route
        df_route_id['catchment {}'.format(tag)] = df_route_id['catchment {}'.format(tag)].fillna(0)
        df_route_id['catchment {}'.format(tag)] = df_route_id['catchment {}'.format(tag)].apply(lambda x: round(x, 0)) 

        df_route_type['catchment {}'.format(tag)] = res_mode
        df_route_type['catchment {}'.format(tag)] = df_route_type['catchment {}'.format(tag)].fillna(0)
        df_route_type['catchment {}'.format(tag)] = df_route_type['catchment {}'.format(tag)].apply(lambda x: round(x, 0)) 

# %%
df_route_id

# %% [markdown]
# # Frequency

# %%
if 'departures' in links.columns:
    links['link_has_timetable'] = links['departures'].apply(lambda x: 0 if x is None else 1)
    lines_with_timetable = links.groupby('route_id')['link_has_timetable'].min()
    links['line_has_timetable'] = links['route_id'].map(lines_with_timetable)
    links.drop(columns='link_has_timetable', inplace=True)
else:
    links['line_has_timetable'] = 0

# %%
from datetime import datetime

def time_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def retrieve_avg_headway(departures):
    deps_seconds = [time_seconds(t) for t in departures]
    if len(deps_seconds) >= 1:
        gaps = [deps_seconds[i] - deps_seconds[i-1] for i in range(1, len(deps_seconds))]
        return int(np.average(gaps))
    return None

def retrieve_oph_headway(departures):
    deps_seconds = [time_seconds(t) for t in departures]
    if len(deps_seconds) >= 1:
        gaps = [deps_seconds[i] - deps_seconds[i-1] for i in range(1, len(deps_seconds))]
        return max(gaps)
    return None

def retrieve_ph_headway(departures):
    deps_seconds = [time_seconds(t) for t in departures]
    if len(deps_seconds) >= 1:
        gaps = [deps_seconds[i] - deps_seconds[i-1] for i in range(1, len(deps_seconds))]
        return min(gaps)
    return None

def retrieve_service_hours(departures):
    deps_seconds = [time_seconds(t) for t in departures]
    if len(deps_seconds) >= 1:
        return (deps_seconds[-1] - deps_seconds[0])/3600

# %%
links.loc[links['line_has_timetable'] == 1, 'frequency_per_day'] = links.loc[links['line_has_timetable'] == 1].apply(lambda x: len(x['departures']), 1)
links.loc[links['line_has_timetable'] == 0, 'frequency_per_day'] = links.loc[links['line_has_timetable'] == 0].apply(lambda x: np.ceil(3600*(x['nb_peak_hours']/x['headway_ph'] + (x['nb_service_hours'] - x['nb_peak_hours'])/x['headway_oph'])), 1)

# %%
links.loc[links['line_has_timetable']==1, 'headway'] = links.loc[links['line_has_timetable']==1, 'departures'].apply(retrieve_avg_headway)
links.loc[links['line_has_timetable']==1, 'headway_ph'] = links.loc[links['line_has_timetable']==1, 'departures'].apply(retrieve_ph_headway)
links.loc[links['line_has_timetable']==1, 'headway_oph'] = links.loc[links['line_has_timetable']==1, 'departures'].apply(retrieve_oph_headway)
links.loc[links['line_has_timetable']==1, 'nb_service_hours'] = links.loc[links['line_has_timetable']==1, 'departures'].apply(retrieve_service_hours)
links.loc[links['line_has_timetable']==1, 'nb_peak_hours'] = None

# %%
links['frequency (veh/hour)'] = 1/links['headway']*3600
links['frequency ph (veh/hour)'] = 1/links['headway_ph']*3600
links['frequency oph (veh/hour)'] = 1/links['headway_oph']*3600

# %%
links['headway'] = links.apply(lambda x: x['headway'] if (x['headway_ph'] == x['headway_oph']) else None, 1)
links['frequency (veh/hour)'] = links.apply(lambda x: x['frequency (veh/hour)'] if (x['frequency ph (veh/hour)'] == x['frequency oph (veh/hour)']) else None, 1)

# %%
res_hour = (links.groupby('route_id')['frequency (veh/hour)'].agg('mean')).to_dict()
res_day = (links.groupby('route_id')['frequency_per_day'].agg('mean')).to_dict()

df_route_id['frequency (veh/day)'] = res_day
df_route_id['frequency (veh/hour)'] = res_hour

if display_ph_columns:
    res_ph = (links.groupby('route_id')['frequency ph (veh/hour)'].agg('mean')).to_dict()
    res_oph = (links.groupby('route_id')['frequency oph (veh/hour)'].agg('mean')).to_dict()

    df_route_id['frequency ph (veh/hour)'] = res_ph
    df_route_id['frequency oph (veh/hour)'] = res_oph

# %%
res_hour = (links.groupby('trip_id')['frequency (veh/hour)'].agg('mean')).to_dict()
res_day = (links.groupby('trip_id')['frequency_per_day'].agg('mean')).to_dict()

df_trip_id['frequency (veh/day)'] = res_day
df_trip_id['frequency (veh/hour)'] = res_hour

if display_ph_columns:
    res_ph = (links.groupby('trip_id')['frequency ph (veh/hour)'].agg('mean')).to_dict()
    res_oph = (links.groupby('trip_id')['frequency oph (veh/hour)'].agg('mean')).to_dict()

    df_trip_id['frequency ph (veh/hour)'] = res_ph
    df_trip_id['frequency oph (veh/hour)'] = res_oph

# %%
res_hour = (links.groupby('route_type')['frequency (veh/hour)'].agg('mean')).to_dict()
res_day = (links.groupby('route_type')['frequency_per_day'].agg('mean')).to_dict()

df_route_type['frequency (veh/day)'] = res_day
df_route_type['frequency (veh/hour)'] = res_hour

# %% [markdown]
# # Line Length

# %%
# preparation. if length is NaN, or if shape dist travel exist.

length_col = None
if 'length' in links.columns and length_col == None:
    if len(links[links['length'].isnull()])==0:
        length_col = 'length'
        
if 'shape_dist_traveled' in links.columns and length_col == None:
    if len(links[links['shape_dist_traveled'].isnull()])==0:
        length_col = 'shape_dist_traveled'

if length_col == None:
    print('create length from geometry')
    links['length'] = links.to_crs(crs).length
    length_col = 'length'

# %% [markdown]
# # Number of station per line

# %%
# o-->o-->o-->o and  o<--o<--o<--o
# est-ce que j'ai 8 ou 4 stations ?
# j'ai 4 stations par trip et 4 stations par route (si c'est les memes).
# comment savoir si cest les mêmes : clustering?
# pour l'instant on prend tous les noeuds unique par route_id ou route_type (col='route_id', route_id)
def get_num_station(col='route_id'):
    link = links.groupby(col)[['a','b']].agg({'a':set,'b':set})
    link['node_len'] = link.apply(lambda row: len(row['a'].union(row['b'])), axis=1)
    return link['node_len'].to_dict()

# %%
links['a_name'] = links['a'].map(nodes['stop_name'].to_dict())
links['b_name'] = links['b'].map(nodes['stop_name'].to_dict())

if len(nodes['stop_name'].values.tolist()) > len(nodes['stop_name'].unique()):
    print('!! Duplicates in node names !!')

# %%
dict_nb_trips = links[['route_id', 'trip_id']].drop_duplicates().groupby('route_id')['trip_id'].count().to_dict()
df_route_id['type'] = df_route_id.index.map(dict_nb_trips)
df_route_id['type'] = df_route_id['type'].apply(lambda x: 'circular' if x == 1 else 'linear')

# %%
def get_node_sequence(route_id, trip_or_route = 'route'):
    if trip_or_route == 'route':
        if df_route_id.loc[route_id]['type'] == 'linear':
            route_id = route_id + '_0'
    links_route = links.loc[links.trip_id == route_id]
    links_route = links_route.sort_values(by='link_sequence')
    nodes_seq = []
    for i in range(len(links_route)):
        nodes_seq += [links_route.iloc[i]['a']]
    nodes_seq += [links_route.iloc[-1]['b']]
    return nodes_seq

# %%
nodes_stops = dict(zip(nodes.index, nodes['stop_name']))

def get_stops_sequence(route_id, trip_or_route='route'):
    nodes_seq = get_node_sequence(route_id, trip_or_route = trip_or_route)
    stops_seq = []
    for node in nodes_seq:
        stops_seq += [nodes_stops[node]]
    return stops_seq

# %%
df_route_id['stations sequence'] = [get_stops_sequence(route_id) for route_id in df_route_id.index]
df_route_id['nodes sequence'] = [get_node_sequence(route_id) for route_id in df_route_id.index]
df_route_id['nb stations'] = df_route_id['stations sequence'].apply(lambda x: len(x))

# %%
df_trip_id['stations sequence'] = [get_stops_sequence(trip_id, trip_or_route='trip') for trip_id in df_trip_id.index]
df_trip_id['nodes sequence'] = [get_node_sequence(trip_id, trip_or_route='trip') for trip_id in df_trip_id.index]
df_trip_id['nb stations'] = df_trip_id['stations sequence'].apply(lambda x: len(x))

# %%
stations_route_type = pd.DataFrame(df_route_id.groupby('route_type')['stations sequence'].agg(lambda x: list(set(sum(x, [])))))
stations_route_type['nb stations'] = stations_route_type['stations sequence'].apply(lambda x: len(x))
df_route_type = df_route_type.merge(stations_route_type, left_on=df_route_type.index, right_on=stations_route_type.index, how='left')
df_route_type = df_route_type.rename(columns={'key_0': 'route_type'})
df_route_type = df_route_type.set_index('route_type')

# %% [markdown]
# ## Connections

# %%
iterable = list(zip(nodes['stop_name'], nodes.index))

stops_nodes = defaultdict(set)
for key, value in iterable:
    stops_nodes[key].add(value)
stops_nodes = dict(stops_nodes)

# %%
iterable = list(zip(links['a'], links['route_id']))
iterable = iterable + list(zip(links['b'], links['route_id']))

nodes_routes = defaultdict(set)
for key, value in iterable:
    nodes_routes[key].add(value)
nodes_routes = dict(nodes_routes)

# %%
stops_routes = {}

for stop, node_list in stops_nodes.items():
    routes = set()
    for node in node_list:
        if node in nodes_routes:
            routes.update(nodes_routes[node])
    stops_routes[stop] = routes

# %%
hubs = pd.DataFrame.from_dict(stops_routes, orient='index')
hubs['lines'] = hubs.apply(lambda row: [val for val in row if pd.notnull(val)], axis=1)
hubs = hubs.drop(columns=[i for i in range(len(hubs.columns) - 1)])
hubs['nb_lines'] = hubs['lines'].apply(lambda x: len(x))
hubs = hubs.sort_values(by='nb_lines', ascending=False)

# %%
dict_route_type = dict(zip(df_route_id.index, df_route_id['route_type']))
dict_veh = dict(zip(df_route_id['route_type'], df_route_id['veh_capacity (PAX)']))
route_order = sorted(dict_veh, key=lambda x: int(dict_veh[x]), reverse=True)

def lines_to_dict(lines):
    route_dict = {route_type: [] for route_type in route_order}
    for line in lines:
        route_type = dict_route_type.get(line)
        if route_type in route_dict:
            route_dict[route_type].append(line)
    route_dict = {k: sorted(v) for k, v in route_dict.items() if v}
    return route_dict

hubs['lines'] = hubs['lines'].apply(lines_to_dict)

# %%
from shapely.ops import unary_union

def centroid(geometries):
    combined_geometry = unary_union(geometries)
    return combined_geometry.centroid

centroids = pd.DataFrame(nodes.groupby('stop_name')['geometry'].agg(centroid))
hubs = hubs.merge(centroids, left_on=hubs.index, right_on=centroids.index, how='left')

hubs = hubs.rename(columns={'key_0': 'stop_name'})
hubs = hubs.set_index('stop_name')

# %%
# hubs['stop_radius'] = hubs['lines'].apply(lambda x: max(catchment_radius[mode] for mode in x.keys()))
hubs['stop_radius'] = default_catchment_radius

# %%
def get_connections(row):
    route_id = row.name
    connections = set()
    for station in row['stations sequence']:
        if station in stops_routes:
            connections.update(stops_routes[station])
    connections.discard(route_id)  # Supprimer la route_id de l'ensemble des connexions
    return lines_to_dict(connections), len(connections)

df_route_id[['connexions', 'nb lines connected']] = df_route_id.apply(lambda row: pd.Series(get_connections(row)), axis=1)

# df_route_id[['connexions']].loc[df_route_id['connexions'] == 'tertiary']

# %% [markdown]
# # Operating costs

# %%
#frequency = freq moy jour
## TODO: veh_km ? et pas veh_km/h
## TODO: change field name capex to opex
def get_veh_kmh(col='route_id', length_col='length'):
    link = links.groupby([col, 'trip_id'])[[length_col, 'frequency_per_day', 'nb_service_hours']].agg({length_col:'sum', 'frequency_per_day': 'mean', 'nb_service_hours': 'mean'})
    link['veh_km/h'] = np.ceil(link['frequency_per_day'] * link[length_col]) / 1000 / link['nb_service_hours'] #to km/H
    return link.reset_index().groupby(col)['veh_km/h'].agg('sum').to_dict()

def get_veh_km(col='route_id', length_col='length'):
    link = links.groupby([col, 'trip_id'])[[length_col, 'frequency_per_day', 'nb_service_hours']].agg({length_col:'sum', 'frequency_per_day': 'mean', 'nb_service_hours': 'mean'})
    link['veh_km/h'] = np.ceil(link['frequency_per_day'] * link[length_col]) / 1000  #to km/H
    return link.reset_index().groupby(col)['veh_km/h'].agg('sum').to_dict()

# %%
res = get_veh_km('route_id', 'length')
df_route_id['veh_km_day'] = res
df_route_id['veh_km_year'] = df_route_id['veh_km_day'] * coef_day_to_year

res = links.groupby('route_id')['capex'].agg('mean').to_dict()
df_route_id['capex'] = res

df_route_id['vehicle_cost_km_day'] = df_route_id['veh_km_day'] * df_route_id['capex']
df_route_id['vehicle_cost_km_year'] = df_route_id['vehicle_cost_km_day'] * coef_day_to_year

# %% [markdown]
# # Speed

# %%
if 'google_speed' in rlinks.columns:
    rlinks['speed'] = rlinks['google_speed']
    rlinks['time'] = rlinks['google_time']
elif 'here_speed' in rlinks.columns:
    rlinks['speed'] = rlinks['here_speed']
    rlinks['time'] = rlinks['here_time']

rlinks['speed'] = rlinks['speed'].astype(float)
rlinks['time'] = rlinks['time'].astype(float)

# %%
if ('road_link_list' in links.columns) and use_road_network:
    line_links = links[['trip_id', 'road_link_list']]
    road_link_list = line_links.explode('road_link_list').set_index('road_link_list')
    road_link_list = road_link_list.merge(rlinks[['time', 'length', 'speed', 'osm_highway', 'geometry']], left_index=True, right_index=True, how='left')
    road_link_list = gpd.GeoDataFrame(road_link_list.loc[road_link_list['osm_highway'] != 'train'], geometry='geometry', crs='EPSG:4326')
    road_link_list.drop_duplicates(inplace=True)

    res_times = road_link_list.groupby('trip_id')['time'].agg(np.nansum).to_dict()
    df_trip_id['time'] = res_times

    res_lengths = road_link_list.groupby('trip_id')['length'].agg(np.nansum).to_dict()
    df_trip_id['length'] = res_lengths

else:
    res_times = links.groupby('trip_id')['time'].agg(np.nansum).to_dict()
    df_trip_id['time'] = res_times

    res_lengths = links.groupby('trip_id')['length'].agg(np.nansum).to_dict()
    df_trip_id['length'] = res_lengths

# %%
## Time reduction parameters:
res_inter_stop = links.groupby('trip_id')['stop_interval'].agg(np.nanmean).to_dict()
df_trip_id['stop_interval'] = res_inter_stop
res_road_pt_factor = links.groupby('trip_id')['road_pt_factor'].agg(np.nanmean).to_dict()
df_trip_id['road_pt_factor'] = res_road_pt_factor

# Computation
df_trip_id['length (km)'] = df_trip_id['length']/1000

df_trip_id['time raw (min)'] = df_trip_id['time']/60
df_trip_id['time (min)'] = (df_trip_id['time']*df_trip_id['road_pt_factor'] + df_trip_id.apply(lambda x: max(0, x['nb stations']-2)*x['stop_interval'], axis=1))/60

#df_trip_id['speed_raw'] = df_trip_id['length']/df_trip_id['time']*3.6
df_trip_id['speed (km/h)'] = df_trip_id['length (km)']/(df_trip_id['time (min)']/60)

df_trip_id.drop(columns=['stop_interval', 'road_pt_factor'], inplace=True)

# %%
res = df_trip_id.groupby('route_id')['length'].agg(np.average).to_dict()
df_route_id['length (km)'] = res
df_route_id['length (km)'] /= 1000

res = df_trip_id.groupby('route_id')['time (min)'].agg(np.average).to_dict()
df_route_id['time (min)'] = res

res = df_trip_id.groupby('route_id')['time (min)'].agg(np.nansum).to_dict()
df_route_id['round trip time (s)'] = res
df_route_id['round trip time (s)'] *= 60

df_route_id['speed (km/h)'] = df_route_id['length (km)'] / (df_route_id['time (min)']/60)

# %% [markdown]
# ## Operational Fleet

# %%
def get_fleet_frequency(route_id):
    time_min = df_route_id.loc[route_id, 'time (min)']
    try:
        freq_ph = df_route_id.loc[route_id, 'frequency ph (veh/hour)']
    except KeyError:
        freq_ph = df_route_id.loc[route_id, 'frequency (veh/hour)']
    return np.ceil(2 * freq_ph * time_min / 60)

# %%
def get_fleet_timetable(route_id):
    link = links.loc[links['route_id'] == route_id]
    stations_sequence = df_route_id.loc[route_id]['nodes sequence']
    circular_line = df_route_id.loc[route_id]['type'] == 'circular'

    termini = [stations_sequence[0], stations_sequence[-1]]
    link['a_terminus'] = link['a'].isin(termini)
    link['b_terminus'] = link['b'].isin(termini)

    if not circular_line:
        dep0_times = link.loc[(link['trip_id'] == route_id + '_0') & link['a_terminus']]['departures'].values.tolist()[0]
        dep1_times = link.loc[(link['trip_id'] == route_id + '_1') & link['a_terminus']]['departures'].values.tolist()[0]
        arr0_times = link.loc[(link['trip_id'] == route_id + '_0') & link['b_terminus']]['arrivals'].values.tolist()[0]
        arr1_times = link.loc[(link['trip_id'] == route_id + '_1') & link['b_terminus']]['arrivals'].values.tolist()[0]

        dep0 = [time_seconds(t) for t in dep0_times]
        dep1 = [time_seconds(t) for t in dep1_times]
        arr0 = [time_seconds(t) for t in arr0_times]
        arr1 = [time_seconds(t) for t in arr1_times]

        travel_time0 = arr0[0] - dep0[0]
        travel_time1 = arr1[0] - dep1[0]

        nb_interbuses = []

        for bus in dep0:
            departure = bus
            arrival = departure + travel_time0
            if arrival < max(dep1):
                departure_ = min([dep for dep in dep1 if dep>arrival])
                tmax = departure_ + travel_time1
            else:
                tmax = max([max(dep0), max(arr1)])

            t = bus
            necessary_buses = 1
            reserve_buses = 0
            departures = [dep for dep in dep0 if dep>t]
            arrivals = [arr for arr in arr1 if arr>t]

            while t < tmax:
                try:
                    next_event = min([min(departures), min(arrivals)])
                except ValueError:
                    break
                t=next_event
                if (next_event in departures) and (next_event in arrivals):
                    departures.pop(0)
                    arrivals.pop(0)
                    pass
                elif next_event in departures:
                    departures.pop(0)
                    if reserve_buses >= 1:
                        reserve_buses -= 1
                    else:
                        necessary_buses += 1
                elif next_event in arrivals:
                    arrivals.pop(0)
                    necessary_buses += 1
                    reserve_buses += 1

            nb_interbuses.append(necessary_buses)

        for bus in dep1:
            departure = bus
            arrival = departure + travel_time1
            if arrival < max(dep0):
                departure_ = min([dep for dep in dep0 if dep>arrival])
                tmax = departure_ + travel_time0
            else:
                tmax = max([max(dep1), max(arr0)])

            t = bus
            necessary_buses = 1
            reserve_buses = 0
            departures = [dep for dep in dep1 if dep>t]
            arrivals = [arr for arr in arr0 if arr>t]

            while t < tmax:
                try:
                    next_event = min([min(departures), min(arrivals)])
                except ValueError:
                    break
                t=next_event
                if (next_event in departures) and (next_event in arrivals):
                    departures.pop(0)
                    arrivals.pop(0)
                    pass
                elif next_event in departures:
                    departures.pop(0)
                    if reserve_buses >= 1:
                        reserve_buses -= 1
                    else:
                        necessary_buses += 1
                elif next_event in arrivals:
                    arrivals.pop(0)
                    necessary_buses += 1
                    reserve_buses += 1

            nb_interbuses.append(necessary_buses)
    
    else:
        dep0_times = link.loc[(link['trip_id'] == route_id + '_0') & link['a_terminus']]['departures'].values.tolist()[0]
        arr0_times = link.loc[(link['trip_id'] == route_id + '_0') & link['b_terminus']]['arrivals'].values.tolist()[0]

        dep0 = [time_seconds(t) for t in dep0_times]
        arr0 = [time_seconds(t) for t in arr0_times]

        travel_time0 = arr0[0] - dep0[0]

        nb_interbuses = []
        for bus in dep0:
            departure = bus
            arrival = departure + travel_time0
            n_buses = len([dep for dep in dep0 if (dep>=bus and dep<arrival)])
            nb_interbuses.append(n_buses)

    return max(max(nb_interbuses), 2)

# %%
def get_service_hours_timetable(route_id):
    link = links.loc[links['route_id'] == route_id]
    stations_sequence = df_route_id.loc[route_id]['nodes sequence']
    circular_line = df_route_id.loc[route_id]['type'] == 'circular'

    termini = [stations_sequence[0], stations_sequence[-1]]
    link['a_terminus'] = link['a'].isin(termini)
    link['b_terminus'] = link['b'].isin(termini)

    if not circular_line:
        dep0_times = link.loc[(link['trip_id'] == route_id + '_0') & link['a_terminus']]['departures'].values.tolist()[0]
        dep1_times = link.loc[(link['trip_id'] == route_id + '_1') & link['a_terminus']]['departures'].values.tolist()[0]
        arr0_times = link.loc[(link['trip_id'] == route_id + '_0') & link['b_terminus']]['arrivals'].values.tolist()[0]
        arr1_times = link.loc[(link['trip_id'] == route_id + '_1') & link['b_terminus']]['arrivals'].values.tolist()[0]

        dep0 = [time_seconds(t) for t in dep0_times]
        dep1 = [time_seconds(t) for t in dep1_times]
        arr0 = [time_seconds(t) for t in arr0_times]
        arr1 = [time_seconds(t) for t in arr1_times]

        time_range = np.ceil((max([max(arr0), max(arr1)]) - min([min(dep0), min(dep1)]))/3600)

    return time_range

# %%
links.loc[links['line_has_timetable']==1, 'fleet'] = links.loc[links['line_has_timetable']==1, 'route_id'].apply(get_fleet_timetable)
links.loc[links['line_has_timetable']==1, 'nb_service_hours'] = links.loc[links['line_has_timetable']==1, 'route_id'].apply(get_service_hours_timetable)

links.loc[links['line_has_timetable']==0, 'fleet'] = links.loc[links['line_has_timetable']==0, 'route_id'].apply(get_fleet_frequency)

# %%
res = (links.groupby('trip_id')['nb_service_hours'].agg('mean')).to_dict()
df_trip_id['nb_service_hours'] = res

# %%
res = (links.groupby('route_id')['fleet'].agg('mean')).to_dict()
df_route_id['fleet'] = res

res = (links.groupby('route_id')['nb_service_hours'].agg('mean')).to_dict()
df_route_id['nb_service_hours'] = res

df_route_id['frequency (veh/hour)'] = df_route_id['frequency (veh/day)'] / df_route_id['nb_service_hours']

# %%
res = df_route_id.reset_index().groupby('route_type')['fleet'].agg('sum').to_dict()
df_route_type['fleet'] = res

res = df_route_id.reset_index().groupby('route_type')['nb_service_hours'].agg('max').to_dict()
df_route_type['nb_service_hours'] = res

# %% [markdown]
# ## Modal splits

# %% [markdown]
# ## Desserte

# %%
if od_file_provided:
    
    nodes_with_zones_ = nodes_with_zones.copy().drop(columns='trip_ids').explode('route_ids')

    if 'nom_com' in zonage.columns:
        zone_to_com = zonage.set_index('zone_id')['nom_com'].to_dict()
        nodes_with_zones['nom_com'] = nodes_with_zones['zone_id'].map(zone_to_com)
        nodes_with_zones_['nom_com'] = nodes_with_zones_['zone_id'].map(zone_to_com)
        serving_com = nodes_with_zones_.groupby('route_ids')['nom_com'].agg(set).to_dict()
        df_route_id['served_coms'] = df_route_id.index.map(serving_com)
        od['origin_com'] = od['origin'].map(zone_to_com)
        od['destination_com'] = od['destination'].map(zone_to_com)

    serving_zone = nodes_with_zones_.groupby('route_ids')['zone_id'].agg(set).to_dict()
    df_route_id['served_zones'] = df_route_id.index.map(serving_zone)

# %% [markdown]
# ## Parts modales

# %%
if od_file_provided:

    if 'origin_com' in od.columns:
        origin_field, destination_field, serving = 'origin_com', 'destination_com', serving_com
    else: 
        origin_field, destination_field, serving = 'origin', 'destination', serving_zone

    if od_file_provided:
        volumes = [col for col in od.columns if 'volume_' in col]
        if volumes:
            dic_total_vol = {}
            dic_vol = {}
            dic_modal_share = {}
            for line, zones in serving.items():
                mode = df_route_id.loc[line]['route_type']
                relevant_ods = od.copy()
                if len(zones) >= 2:
                    relevant_ods = relevant_ods.loc[relevant_ods[origin_field].isin(zones) & relevant_ods[destination_field].isin(zones) & (relevant_ods[origin_field] != relevant_ods[destination_field])]
                else:
                    relevant_ods = relevant_ods.loc[relevant_ods[origin_field].isin(zones) & relevant_ods[destination_field].isin(zones)]

                try:
                    dic_total_vol[mode][line] = relevant_ods['volume'].sum()
                except KeyError:
                    dic_total_vol[mode] = {}
                    dic_total_vol[mode][line] = relevant_ods['volume'].sum()
                if 'volume_{}'.format(mode) in volumes:
                    try:
                        dic_vol[mode][line] = relevant_ods['volume_{}'.format(mode)].sum()
                        dic_modal_share[mode][line] = dic_vol[mode][line] / dic_total_vol[mode][line]  
                    except KeyError:
                        dic_vol[mode] = {}
                        dic_vol[mode][line] = relevant_ods['volume_{}'.format(mode)].sum()
                        dic_modal_share[mode] = {}
                        dic_modal_share[mode][line] = dic_vol[mode][line] / dic_total_vol[mode][line]  

            for mode in dic_modal_share:
                df_route_id.loc[df_route_id['route_type'] == mode, 'modal share'] = df_route_id.loc[df_route_id['route_type'] == mode].index.map(dic_modal_share[mode])
                df_route_type.loc[mode, 'modal share OD'] = sum(list(dic_vol[mode].values())) / sum(list(dic_total_vol[mode].values()))

# %% [markdown]
# ## Flux TC

# %%
if od_file_provided:
    modal_shares = links.groupby('route_type')['modal_share_input'].agg(np.nanmean).to_dict()

# %%
if od_file_provided:

    if 'origin_com' in od.columns:
        origin_field, destination_field, serving = 'origin_com', 'destination_com', serving_com
    else: 
        origin_field, destination_field, serving = 'origin', 'destination', serving_zone

    if od_file_provided:
        dic_total_vol_sans_filtre = {}
        dic_modal_shares = {}
        for line, zones in serving.items():

            modal_share = modal_shares[links.loc[links['route_id'] == line, 'route_type'].values.tolist()[0]]
            dic_modal_shares[line] = modal_share
            
            relevant_ods = od.copy()
            if len(zones) >= 2:
                relevant_ods = relevant_ods.loc[relevant_ods[origin_field].isin(zones) & relevant_ods[destination_field].isin(zones) & (relevant_ods[origin_field] != relevant_ods[destination_field])]
            else:
                relevant_ods = relevant_ods.loc[relevant_ods[origin_field].isin(zones) & relevant_ods[destination_field].isin(zones)]

            dic_total_vol_sans_filtre[line] = relevant_ods['volume'].sum()*modal_share 

        df_route_id['modal_share_input'] = dic_modal_shares
        df_route_id['volume'] = dic_total_vol_sans_filtre

# %%
## TODO: il faudra trouver un moyen ou un autre pour supprimer cette partie train, ou de la skip si elle n'est pertinente que dans certains contextes.

if od_file_provided:
    if 'is_train' in od.columns:
        if 'origin_com' in od.columns:
            origin_field, destination_field, serving = 'origin_com', 'destination_com', serving_com
        else: 
            origin_field, destination_field, serving = 'origin', 'destination', serving_zone

        if od_file_provided:
            dic_total_vol = {}
            for line, zones in serving.items():

                modal_share = modal_shares[links.loc[links['route_id'] == line, 'route_type'].values.tolist()[0]]
                dic_modal_shares[line] = modal_share

                relevant_ods = od.copy()
                if len(zones) >= 2:
                    relevant_ods = relevant_ods.loc[~relevant_ods['is_train'] & relevant_ods[origin_field].isin(zones) & relevant_ods[destination_field].isin(zones) & (relevant_ods[origin_field] != relevant_ods[destination_field])]
                else:
                    relevant_ods = relevant_ods.loc[relevant_ods[origin_field].isin(zones) & relevant_ods[destination_field].isin(zones)]

                dic_total_vol[line] = relevant_ods['volume'].sum()*modal_share 

            df_route_id['capturable volume (no train)'] = dic_total_vol

# %% [markdown]
# # Export results

# %% [markdown]
# ## Tables

# %% [markdown]
# ### Characterstics:

# %%
def sec_to_duree(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if seconds >= 30:
        minutes += 1
    if minutes == 60:
        hours += 1
        minutes = 0
    time_str = ""
    if hours > 0:
        time_str += f"{hours}h "
    if minutes > 0:  
        time_str += f"{minutes}min "
    return time_str.strip()

df_route_id["round_trip_time"] = df_route_id["round trip time (s)"].apply(sec_to_duree)

# %%
columns = ['route_type', 
           'nb stations', 
           'length (km)',
           'round_trip_time',
           'speed (km/h)', 
           'frequency (veh/day)', 
           'veh_capacity (PAX)', 
           'fleet',
           'veh_km_year', 
           'vehicle_cost_km_year',
           'geometry']

# %%
char_table = df_route_id.copy()[columns]
char_table['length (km)'] = char_table['length (km)'].apply(np.round, decimals=1)
char_table['speed (km/h)'] = char_table['speed (km/h)'].apply(np.round, decimals=0)
char_table['veh_km_year'] = char_table['veh_km_year'].apply(np.round, decimals=0)

# %%
char_table.drop(columns='geometry').to_csv(output_folder + 'lines_chacteristics.csv')

char_table_geo = gpd.GeoDataFrame(char_table, geometry='geometry', crs='EPSG:4326')
char_table_geo.explode().to_file(output_folder + 'lines_chacteristics.geojson', driver='GeoJSON', engine=io_engine)

# %% [markdown]
# ### Catchment:

# %%
catchment_columns = [col for col in df_route_id.columns if 'catchment' in col]
catchment_columns_ = catchment_columns + ['geometry']

# %%
catch_table = df_route_id.copy()[catchment_columns_]
for col in catchment_columns:
    catch_table[col] = catch_table[col].astype(int)

# %%
catch_table.drop(columns='geometry').to_csv(output_folder + 'lines_catchment.csv')

catch_table_geo = gpd.GeoDataFrame(catch_table, geometry='geometry', crs='EPSG:4326')
catch_table_geo.explode().to_file(output_folder + 'lines_catchment.geojson', driver='GeoJSON', engine=io_engine)

# %%
#TODO : formater les tableaux de sortie de df_route_id ==> caractéristiques / accessibilité / réponse au besoin

# Dans réponse au besoin : estimation du volume de flux TC desservis sans correspondance pour une PM de 20%, estimation du taux de remplissage TC sans correspondance pour une PM TC de 10%

#TODO: ajouter les tableaux globaux df_route_type (longueur totale, nombre de stations, flotte, veh.km/jour, capex/jour, veh.km/an, capex/an) et hubs

# %%
# round numbers
#TODO : change label catchment
# for col in ['catchment population', 'frequency (veh/hours)','length (m)','veh.km/h','round trip time (s)']:
#     df_route_id[col] = df_route_id[col].apply(lambda x :np.round(x,2))
#     df_route_id[col] = df_route_id[col].apply(lambda x :np.round(x,2))

# %%
#df_route_id = df_route_id.fillna('null')
#df_route_type = df_route_type.fillna('null')

# %%
df_route_id['length (km)'] = df_route_id['length (km)'].apply(np.round, decimals=1)
df_route_id['speed (km/h)'] = df_route_id['speed (km/h)'].apply(np.round, decimals=1)
df_route_id['veh_km_year'] = df_route_id['veh_km_year'].apply(np.round, decimals=0)

# %%
df_route_id.drop(columns=['geometry', 'nodes sequence']).to_csv(output_folder + 'route_id_metrics.csv')
# df_route_id

# %%
df_route_type.drop(columns='stations sequence').to_csv(output_folder + 'route_type_metrics.csv')
# df_route_type

# %% [markdown]
# ## Modal share table:

# %%
if od_file_provided:
    
    modal_table = df_route_id[['modal share', 'volume', 'capturable volume (no train)', 'geometry']]
    modal_table['modal share (%)'] = modal_table['modal share'].apply(lambda x: np.round(100*x, 1))
    modal_table['volume'] = modal_table['volume'].apply(np.round, decimals=0)
    modal_table['capturable volume (no train)'] = modal_table['capturable volume (no train)'].apply(np.round, decimals=0)

    modal_table.drop(columns=['geometry', 'modal share']).to_csv(output_folder + 'line_flows.csv')

    modal_table = gpd.GeoDataFrame(modal_table, geometry='geometry', crs='EPSG:4326')
    modal_table.explode().to_file(output_folder + 'line_flows.geojson', driver='GeoJSON', engine=io_engine)

# %%
## TODO: add results for timetable driver lines ?

# %% [markdown]
# ## Geomatic outputs

# %% [markdown]
# Hubs

# %%
hubs_plot = hubs.copy()
hubs_plot['lines'] = hubs_plot['lines'].apply(lambda x: str(x).replace(',', ';').replace("'", '')[1:-1])
hubs_plot.drop(columns='geometry').to_csv(output_folder + 'hubs.csv')

hubs = gpd.GeoDataFrame(hubs, geometry='geometry', crs='EPSG:4326')
hubs.to_file(output_folder + 'hubs.geojson', driver='GeoJSON', engine=io_engine)

# %% [markdown]
# Common sections

# %%
# Renvoie un fichier geojson avec les tronçons en commun entre plusieurs lignes

# %%
# clustering de 500m pour a et b et cluster d'appartenance
coords_nodes = np.array(nodes['geometry'].apply(lambda point: (point.x, point.y)).tolist())

# Convertir 500 mètres en degrés : 111 km = 1 degré de latitude
eps_lat = clustering_radius / (111 * 1000)  # Environ 0.0045 degrés

# 1 degré de longitude dépend de la latitude
mean_latitude = np.mean(coords_nodes[:, 1])
eps_lon = clustering_radius / (111 * 1000 * np.cos(np.radians(mean_latitude)))

# Appliquer DBSCAN avec une distance euclidienne pondérée
db = DBSCAN(eps=1, min_samples=1, metric='euclidean').fit(coords_nodes / [eps_lon, eps_lat])

# Ajouter les labels de cluster au GeoDataFrame
nodes['cluster'] = db.labels_

# %%
nodes = nodes.reset_index()

# %%
links = links.merge(nodes[['index', 'cluster']], left_on='a', right_on='index', how='left').drop(columns='index').rename(columns={'cluster': 'a_clustered'})
links = links.merge(nodes[['index', 'cluster']], left_on='b', right_on='index', how='left').drop(columns='index').rename(columns={'cluster': 'b_clustered'})

# %%
l_troncons = links.groupby(['a_clustered', 'b_clustered'])['route_id'].agg(list).reset_index()
l_troncons['nb_lines'] = l_troncons['route_id'].apply(lambda x: len(x))

# %%
l_troncons_communs = l_troncons[l_troncons.nb_lines > 1.]

# %%
troncons_communs = len(l_troncons_communs) > 0.
print(troncons_communs)

# %%
if troncons_communs:
    l_troncons_communs['geometry'] = l_troncons_communs.apply(
        lambda row : links[(links.a_clustered == row['a_clustered']) & (links.b_clustered == row['b_clustered'])].drop_duplicates(subset=['a_clustered', 'b_clustered'], keep='first').geometry.values[0],
        axis=1
        )
    l_troncons_communs['stations_a'] = l_troncons_communs['a_clustered'].apply(lambda x: list(nodes[nodes.cluster == x]['stop_name'].unique()))
    l_troncons_communs['stations_b'] = l_troncons_communs['b_clustered'].apply(lambda x: list(nodes[nodes.cluster == x]['stop_name'].unique()))
    l_troncons_communs = gpd.GeoDataFrame(l_troncons_communs[['stations_a', 'stations_b', 'route_id', 'nb_lines', 'geometry']], geometry='geometry', crs='EPSG:4326')
    l_troncons_communs.to_file(output_folder + 'pt_common_sections.geojson', driver='GeoJSON', engine=io_engine)

# %% [markdown]
# df_route_id

# %%
gpd.GeoDataFrame(df_route_id, geometry='geometry', crs='EPSG:4326').to_file(output_folder + 'pt_network_kpis.geojson')

# %% [markdown]
# Nodes catchment

# %%
#TODO pcq c'est DYNAMIQUE

# %%
# # Using get catchment : get the catchment radius of each node (get larger one if used by many modes)
for col in catchment_radii:
    suf = col.split('catchment_radius_')[1]
    for density in densities:
        if density == 'density' and 'population_density' not in densities:
            tag = 'population'
        elif density == 'density':
            tag = 'x'
        else:
            tag = density.split('_density')[0]

        mesh, node_dist = meshes[tag], node_dists[tag]

        link = links.groupby('route_type')[['a', 'b', 'route_type', col]].agg({'a': set, 'b': set, 'route_type': 'first', col:np.nanmean})
        link['node'] = link.apply(lambda row: row['a'].union(row['b']), axis=1)
        link = link.drop(columns=['a','b'])
        ## add catchment radius for the route_type
        link = link.explode('node').reset_index(drop=True)
        link = link.sort_values(col,ascending=False).drop_duplicates('node',keep='first')
        link = node_dist.merge(link, left_on='node_index', right_on='node')
        link = link[link['distances'] <= link[col]]

        temp_dict = link.groupby('node_index')[tag].sum().to_dict()
        nodes['catchment {} {}'.format(suf, tag)] = nodes['index'].map(temp_dict.get)

        temp_dict = link.groupby('node_index')[col].agg('first').to_dict() 
        nodes[col] = nodes['index'].map(temp_dict.get)

# %%
nodes.to_file(output_folder + 'nodes.geojson', driver='GeoJSON', engine=io_engine)

# %% [markdown]
# ## Graphs and pictures

# %%
# plot = df_route_type.reset_index().plot(kind='bar', x='route_type', y='catchment', color='#559bb4', rot=0, figsize=[10, 5])
# plot.set_title('Couverture population par mode')
# plot.set_ylabel('')
# plot.set_xlabel("route_type")
# plot.legend([])


