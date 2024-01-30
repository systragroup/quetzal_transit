def main():
    #!/usr/bin/env python
    # coding: utf-8
    
    # In[33]:
    
    
    import sys
    import json
    
    
    params = {'gtfs':'../../gtfs/stl.zip','selected_day':1,'time_range' : ['6:00:00', '8:59:00']}
    
    default = {'training_folder': '../../scenarios/test', 'params':params} # Default execution parameters
    manual, argv = (True, default) if 'ipykernel' in sys.argv[0] else (False, dict(default, **json.loads(sys.argv[1])))
    print(argv)
    import os
    
    os.environ['BUCKET_NAME']='quetzal-api-bucket'
    
    
    # In[30]:
    
    
    import time
    import geopandas as gpd
    import pandas as pd
    sys.path.insert(0, r'../../../quetzal/') # Add path to quetzal
    sys.path.insert(0, r'../../../quetzal/api/GTFS_importer') # Add path to quetzal
    
    from api.GTFS_importer import main
    
    
    # In[ ]:
    
    
    
    
    
    # In[39]:
    
    
    uuid = 'test'
    files = [argv['params']['gtfs']]
    dates = []
    selected_day = argv['params']['selected_day']
    time_range = argv['params']['time_range']
    sm = main.main(uuid, files, dates, selected_day, time_range,export=False)
    
    
    # In[43]:
    
    
    pt_folder = argv['training_folder']+'/inputs/pt/'
    if not os.path.exists(pt_folder):
        os.makedirs(pt_folder)
    
    
    # In[45]:
    
    
    sm.links.to_file(pt_folder+'links.geojson',driver='GeoJSON')
    sm.nodes.to_file(pt_folder+'nodes.geojson',driver='GeoJSON')
    
    
    # In[ ]:
    
    
    
    
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    main()
