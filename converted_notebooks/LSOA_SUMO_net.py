#!/usr/bin/env python
# coding: utf-8
import geopandas as gpd
from bim.sumo import network
import json
from collections import defaultdict
from bim.gis.traffic import od_matrix
import numpy as np


lsoa_boundaries_gdf = gpd.read_file('gmdata.nosync/infuse_lsoa_lyr_2011.shp')

nodes_xml = network.NodesXML()
node_type = "priority"
for i in range(len(lsoa_boundaries_gdf)):
    lsoa_record = lsoa_boundaries_gdf.iloc[i]
    centroid = lsoa_record.geometry.centroid
    nodes_xml.add_node(lsoa_record.geo_code, centroid.x, centroid.y, node_type)

lsoa_boundaries_gdf.loc[:10, 'geometry'].apply(lambda x: list(x.boundary.coords))
nodes_xml.save('gmdata.nosync/lsoa_nodes.nod.xml')

with open('gmdata.nosync/lsoa_connections.json', 'r') as f:
    connected = json.load(f)

edges_xml = network.EdgesXML()
speed = 50
priority = 2
numLanes = 2
for i in range(len(connected)):
    lsoa_from, lsoa_to = connected[i]
    if not lsoa_from or not lsoa_to:
        continue
    centroid = lsoa_record.geometry.centroid
    edges_xml.add_edge(f"edge{i}", lsoa_from, lsoa_to, priority, numLanes, speed)

edges_xml.save('gmdata.nosync/lsoa_edges.edg.xml')

# !netconvert --node-files=gmdata.nosync/lsoa_nodes.nod.xml --edge-files=gmdata.nosync/lsoa_edges.edg.xml \
#   --output-file=gmdata.nosync/lsoa_net.net.xml

## LSOA SUMO routes from OD matrix
connected_dict = {lsoa_code:i for i, lsoa_pair in enumerate(list(map(tuple, connected)))
                  for lsoa_code in lsoa_pair if '' not in lsoa_pair}

edges = defaultdict(set)
for node_from, node_to in connected:
    edges[node_from].add(node_to)
    edges[node_to].add(node_from)


def dfs(node, target_node):
    visited = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n == target_node:
            return True
        for n2 in edges[n]:
            if n2 not in visited:
                stack.append(n2)
                visited.add(n2)
    return False

dfs('E01005126', 'E01005168')

area_od_matrix = od_matrix.AreaOriginDestinationMatrix()
od_gm_df = area_od_matrix.load()
od_gm_df['trip_edge_from'] = od_gm_df.apply(lambda x: connected_dict.get(x['geo_code1'], -1), axis=1)
od_gm_df['trip_edge_to'] = od_gm_df.apply(lambda x: connected_dict.get(x['geo_code2'], -1), axis=1)
od_gm_df = od_gm_df[(od_gm_df['trip_edge_from'] != -1) & (od_gm_df['trip_edge_to'] != -1)].copy()


routes_xml = network.RoutesXML()
depart = 0
for i in range(len(od_gm_df)):
    od_record = od_gm_df.iloc[i]
    edge_from, edge_to = od_record.trip_edge_from, od_record.trip_edge_to
    routes_count = od_record['all']
    departs = depart + np.random.normal(10, 5, routes_count)
    for j in range(routes_count):
        routes_xml.add_trip(f"{i}#{j}", max(0, departs[j]), f"edge{edge_from}", f"edge{edge_to}")

routes_xml.save("gmdata.nosync/lsoa_trips_od.rou.xml")

# !wc -l gmdata.nosync/lsoa_trips_od.rou.xml

list(filter(lambda x: 'E01005126' in x[1], enumerate(connected)))

# !duarouter -n gmdata.nosync/lsoa_net.net.xml -t gmdata.nosync/lsoa_trips_od.rou.xml -o gmdata.nosync/lsoa_od_raw_routes.rou.xml --ignore-errors 2&> gmdata.nosync/duarouter.log')

# ## Search for OD LSOA's missed in LSOA connections
no_od_edge_set = set(od_gm_df[od_gm_df.apply(lambda x: edges.get(x['geo_code1'], -1) ==-1, axis=1)]['geo_code1'])

counties_region_gdf = gpd.read_file("gmdata.nosync/Boundary-line-ceremonial-counties_region.shp")
gm_boundary = counties_region_gdf[counties_region_gdf['NAME'].apply(lambda x: 'manchester' in x.lower())]
gm_boundary = gm_boundary.to_crs(epsg=4326)
gm_boundary.plot()

