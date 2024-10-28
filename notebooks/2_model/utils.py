
from quetzal.engine.pathfinder_utils import  sparse_matrix
from syspy.spatial.spatial import add_geometry_coordinates, nearest
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from numba import jit, njit
import numba as nb
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Literal


# from quetzal_cyclops
def get_epsg(lat: float, lon: float) -> int:
    '''
    lat, lon or y, x
    return EPSG in meter for a given (lat,lon)
    lat is north south 
    lon is est west
    '''
    return int(32700 - round((45 + lat) / 90, 0) * 100 + round((183 + lon) / 6, 0))

# from quetzal_cyclops
def zones_nearest_node(zones,nodes,drop_duplicates=False):
    # getting zones centroids
    centroid = zones.copy()
    centroid['geometry'] = centroid.centroid
    # finding nearest node
    neigh = nearest(centroid, nodes, n_neighbors=1).rename(columns={'ix_one': 'zone_index', 'ix_many': 'node_index'})
    zone_node_dict = neigh.set_index('zone_index')['node_index'].to_dict()
    centroid['node_index'] = centroid.index.map(zone_node_dict.get)
    #print('max_distance found: ', neigh['distance'].max())
    # check for duplicated nodes. if there is. drop the duplicated zones.
    if drop_duplicates:
        if len(centroid.drop_duplicates('node_index')) != len(centroid):
            print('there is zones associates to the same road_node')
            # duplicated = centroid[centroid['node_index'].duplicated()]['node_index'].values
            print('dropping zones: ')
            print(centroid[centroid['node_index'].duplicated()].index.values)
            centroid = centroid.drop_duplicates('node_index')
    return centroid


@jit(nopython=True)
def _unstack(mat):
    # return non inf values in mat as [[row,col,val],[row,col,val]]. so, [o,d,val].
    # pd.DataFrame of this gives us [origin, destination, value] as columns
    row, col = np.where(np.isfinite(mat))
    res = np.zeros((len(row),3))
    for it in nb.prange(len(col)):
        i=row[it]
        j=col[it]
        d=mat[i,j]
        res[it]=[i,j,d]
    return res

def routing(origin, destination, links, weight_col='time', dijkstra_limit=np.inf):
    mat, node_index = sparse_matrix(links[['a', 'b', weight_col]].values)
    index_node = {v: k for k, v in node_index.items()}
    # liste des origines pour le dijkstra
    origin_sparse = [node_index[x] for x in origin]
    origin_dict =  {i:val for i,val in enumerate(origin_sparse)}
    # list des destinations 
    destination_sparse = [node_index[x] for x in destination]
    destination_dict =  {i:val for i,val in enumerate(destination_sparse)}
    # dijktra on the road network from node = incices to every other nodes.
    # from b to a.
    dist_matrix = dijkstra(
        csgraph=mat,
        directed=True,
        indices=origin_sparse,
        return_predecessors=False,
        limit=dijkstra_limit
    )
    # remove non-used destination
    dist_matrix = dist_matrix[:,destination_sparse]
    # unstack amtrix
    dist_matrix = pd.DataFrame(_unstack(dist_matrix),columns=['origin', 'destination', weight_col])
    # rename origin and destination with original indexes.
    dist_matrix['origin'] = dist_matrix['origin'].apply(lambda x: index_node.get(origin_dict.get(x)))
    dist_matrix['destination'] = dist_matrix['destination'].apply(lambda x: index_node.get(destination_dict.get(x)))
    return dist_matrix


def get_catchment_dist(link: gpd.GeoDataFrame, catchment_radius: dict, default: float=500):
    route_type = link['route_type'].unique()
    if len(route_type)>1:
        print('multiple route type for a single route_id.. using first one for catchment radius')
    route_type = route_type[0]
    return catchment_radius.get(route_type, default)


def nearest_radius(one, many, radius=100,to_crs=None):
    try:
        # Assert df_many.index.is_unique
        assert one.index.is_unique
        assert many.index.is_unique
    except AssertionError:
        msg = 'Index of one or many should not contain duplicates'
        print(msg)
        warnings.warn(msg)
    many = add_geometry_coordinates(many, columns=['x_geometry', 'y_geometry'], to_crs=to_crs)
    one = add_geometry_coordinates(one, columns=['x_geometry', 'y_geometry'], to_crs=to_crs)
    
    x = many[['x_geometry', 'y_geometry']].values
    # Fit Nearest neighbors model
    #nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(x)
    nbrs = NearestNeighbors(radius=radius,algorithm='ball_tree').fit(x)

    y = one[['x_geometry', 'y_geometry']].values

    #distances, indices = nbrs.kneighbors(y,return_distance=True)
    distances, indices = nbrs.radius_neighbors(y, radius = radius, return_distance=True)

    indices = pd.DataFrame(indices)
    indices = pd.DataFrame(indices.stack(), columns=['index_nn']).reset_index().rename(
        columns={'level_0': 'ix_one', 'level_1': 'rank'}
    )
    indices['distances'] = distances
    return indices

def create_mesh(zones: gpd.GeoDataFrame ,step: float = 0.01) -> gpd.GeoDataFrame:
    '''
    create a mesh in the zones total bbox at every step (in the units of the zones crs)
    step: degree if crs=4326, else meters. 0.01 deg ~ 1km
    '''
    x_max, y_max = zones.bounds.max()[['maxx','maxy']].values
    x_min, y_min = zones.bounds.min()[['minx','miny']].values

    points = []
    x = x_min
    while x<x_max:
        y = y_min
        while y<y_max:
            points.append(Point(x,y))
            y += step
        x += step
    points = gpd.GeoDataFrame(geometry=points,crs=zones.crs)
    points.index.name='index'
    return points

# https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

@njit()
def fast_point_in_polygon(x: float, y: float , poly: np.ndarray) -> bool:
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in nb.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
        
    return inside


@njit(parallel=True)
def fast_points_in_polygon(points:np.ndarray, polygon:np.ndarray) -> np.ndarray:
    D = np.empty(len(points), dtype=nb.boolean) 
    for i in nb.prange(0, len(points)):
        D[i] = fast_point_in_polygon(points[i,0], points[i,1], polygon)
    return np.where(D)[0]

def points_in_polygon(points:np.ndarray, polygon:gpd.GeoDataFrame) -> np.ndarray:
    '''
    return a list of point in the polygon. values are the index in the points array.
    
    points:np.array[np.array[float,float]]
        list of all the points coords (x,y)
    polygon: gpd.GeoDataFrame
        geodataframe of multiples polygons.
    '''
    try:
        poly = np.array([*polygon.exterior.coords])
        return fast_points_in_polygon(points,poly)
    except:
        res=np.array([])
        #polygon = polygon.geoms
        for pol in polygon.geoms:
            poly = np.array([*pol.exterior.coords])
            val =fast_points_in_polygon(points,poly)
            res = np.append(res,val)
        return res

def population_to_mesh(population: gpd.GeoDataFrame,
                       mesh: gpd.GeoDataFrame = None,
                       step: float = 0.01,
                       col: str = 'population', 
                       fill_missing: Literal['centroid', 'nearest', None] = 'centroid') ->  gpd.GeoDataFrame:
    '''
    create a mesh in the zones total bbox at every step (in the units of the zones crs)
    and assign the population to each node equaly (if 2 node in a zone. they have each 50% of the population)
    population:
        geodataframe with total population by zones ans zones geomerty
     mesh:
        road nodes for example. if None. it will be created with equal step (variable step.)
    step: 
        if mesh is None, Distance between each point degree if crs=4326, else meters. 0.01 deg ~ 1km
    col:
        column name with data to aggregation (population)
    fill_missing: 'centroid', 'nearest', or None
        centroid: zones centroid with no mesh node inside will be added to the mesh
        nearest: zones population with no mesh point inside will be added to the nearest mesh point.
    '''
    import warnings
    warnings.filterwarnings('ignore')
    population=population.copy()
    if population.index.name != 'index':
        population.index.name = 'index'
    # use existing mesh (points .geosjon) or create one.
    if mesh is not None:
        # we need numerical indexes. also,
        # new nodes will be added (new index) for zones with no points inside.
        points = mesh.copy()
        points = points.reset_index(names='node_index')
        points.index.name='index'
    else:
        points = create_mesh(population, step=step)
        
    points_coords = np.array([point.coords[0] for point in points['geometry'].values])
    
    population['nodes'] = population['geometry'].apply(lambda x: points_in_polygon(points_coords,x))
    
    nodes = population.reset_index()[['index','nodes',col]].copy()
    nodes = nodes.explode('nodes').dropna()
    print(len(nodes[nodes['nodes'].duplicated()]),'nodes in multiple zones. will be match to a single zone.')
    
    
    zone_index_dict = nodes.set_index('nodes')['index'].to_dict()
    points['zone'] = points.index.map(zone_index_dict)

    pop_dict = nodes.set_index('nodes')[col].to_dict()
    points[col] = points.index.map(pop_dict)
    points = points.dropna()
    
    # get number of points per zones. divide population equaly between each points
    len_dict = points.groupby('zone')[col].agg(len).to_dict()
    points['num_points'] = points['zone'].apply(lambda x:len_dict.get(x))
    points[col] = points[col] / points['num_points']
    points = points.drop(columns = ['num_points'])
    
    print(len(population) - len(points['zone'].unique()),'unfounded zones')
    
    zones_list = points['zone'].unique()
    unfounded_zones = population.loc[~population.index.isin(zones_list)][['geometry',col]]
    if fill_missing == 'centroid':
        print('Unfound zones centroid will be added to mesh')
        # append unfounded zones centroids as in mesh
        unfounded_zones['geometry'] = unfounded_zones.centroid
        unfounded_zones = unfounded_zones.reset_index().rename(columns={'index':'zone'})
        points = pd.concat([points,unfounded_zones]).reset_index(drop=True)
        points.index.name='index'
    elif fill_missing == 'nearest':
        print('unfound zone will be added to nearest mesh node. zone_index will be lost')
        unfounded_zones = zones_nearest_node(unfounded_zones,points)
        pop_to_append = unfounded_zones.groupby('node_index')[[col]].sum()

        points = points.merge(pop_to_append,left_index=True,right_index=True,how='left')
        points[col+'_y'] = points[col+'_y'].fillna(0)

        points[col] = points[col+'_x'] + points[col+'_y']
        points = points.drop(columns=[col+'_x', col+'_y'])
    else:
        pass
    
    
    points.index.name='index'
    
    return points


def get_acf_distances(nodes: gpd.GeoDataFrame, 
                      mesh: gpd.GeoDataFrame, 
                      crs:int,
                      max_dist: float = 3000) -> gpd.GeoDataFrame:
    '''
    with nearest kneibor in a radius.
    for pt node in nodes, get all mesh nodes in a distance < max_dist
    
    return gpd.Geodateframe with [node_index, mesh_index, distances, population]
    '''

    node_dist = nearest_radius(nodes, mesh, radius=max_dist, to_crs=crs)
    node_dist = node_dist.rename(columns={'ix_one': 'node_index','index_nn':'mesh_index'}).drop(columns='rank')

    nodes_index_dict = nodes.reset_index()['index'].to_dict()
    node_dist['node_index'] = node_dist['node_index'].apply(lambda x: nodes_index_dict.get(x))

    node_dist = node_dist.explode(['mesh_index','distances'])
    population_dict = mesh['population'].to_dict()
    node_dist['population'] = node_dist['mesh_index'].apply(lambda x: population_dict.get(x))
    return node_dist

def get_routing_distances(nodes: gpd.GeoDataFrame, 
                         rnodes: gpd.GeoDataFrame, 
                         rlinks: gpd.GeoDataFrame, 
                         mesh: gpd.GeoDataFrame, 
                         weight_col:str = 'length', 
                         dijkstra_limit: float = np.inf) -> gpd.GeoDataFrame:
    '''
    with dijktra on road network.
    for pt node in nodes, get all mesh nodes in a distance < max_dist. can be change with weight_col
    ex: weight_col = 'time', and dijkstra_limit = 120secs
    
    return gpd.Geodateframe with [node_index, mesh_index, distances, population]
    '''

    # transform PT nodes to nearest road nodes
    node_to_rnode_df = zones_nearest_node(nodes,rnodes)[['node_index']]

    node_rnodes_dict = node_to_rnode_df['node_index'].to_dict()
    rnodes_node_dict = node_to_rnode_df.reset_index().groupby('node_index').agg(list)['index'].to_dict()

    # there may be multiples nodes pointing to the same rnode. so rnodes_node_dict values are lists.
    # need to added them back at the end when we go from rnode to nodes
    origins = list(set(node_rnodes_dict.values()))
    destinations = mesh['node_index'].values
    mat = routing(origins, destinations, rlinks, weight_col=weight_col, dijkstra_limit=dijkstra_limit)

    mat = mat.merge(mesh.reset_index()[['index','node_index','population']],left_on='destination',right_on='node_index',how='left')
    mat = mat.drop(columns=['destination','node_index']).rename(columns={'index':'mesh_index'})

    mat['origin'] = mat['origin'].apply(lambda x: rnodes_node_dict.get(x))
    mat = mat.explode('origin')
    mat = mat.rename(columns={'origin':'node_index', weight_col:'distances'})
    return mat