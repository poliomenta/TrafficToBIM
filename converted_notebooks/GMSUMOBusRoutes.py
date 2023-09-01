#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
import fiona
from bim.sumo.network import SUMOBusStop, BusStopId, BusLaneId, SUMOCoordinate, EdgeId, SUMONetwork
from bim.sumo.network import SUMOBusTrip, BusTripId
from bim.traffic.trips import SUMOTrips

from shapely import geometry
import pandas as pd
import tqdm
import networkx as nx

fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw' # enable KML support which is disabled by default
bus_routes_gdf = gpd.read_file('gmdata.nosync/BusRouteMapData/KML-format/OpenData_BusRoutes.KML')
bus_routes_gdf = bus_routes_gdf.to_crs(27700)
bus_stops_gdf = gpd.read_file('gmdata.nosync/TfGMStoppingPoints.csv')
bus_stops_gdf = bus_stops_gdf[bus_stops_gdf['Status'] == 'act']
bus_top_locations = gpd.points_from_xy(bus_stops_gdf['Easting'], bus_stops_gdf['Northing'])
bus_stops_gdf.set_geometry(bus_top_locations, inplace=True)
bus_stops_gdf.crs = 27700
gdf_routes = bus_routes_gdf.copy()
buffer_distance = 20
gdf_routes['geometry'] = gdf_routes['geometry'].buffer(buffer_distance)
joined_gdf = bus_stops_gdf.sjoin(gdf_routes, how="inner", op="within")
routes_and_stops_gdf = joined_gdf.groupby('Name').apply(lambda x: list(x['AtcoCode'])).rename("bus_stops").to_frame()
routes_and_stops_gdf = routes_and_stops_gdf.reset_index()
routes_and_stops_gdf['bus_service'] = routes_and_stops_gdf['Name'].apply(lambda x: x.split('_')[0])
routes_and_stops_gdf = routes_and_stops_gdf.groupby('bus_service').first().reset_index()

DEFAULT_BUS_STOP_WIDTH = 10

princess_rd_net = SUMONetwork.parse_file('gmdata.nosync/princess_rd_base_net/princess_rd_base_net.json')
edges_df = princess_rd_net.make_edges_df()
edges_df.set_index(edges_df['edge_id'], inplace=True)

net_bounding_box = geometry.box(*edges_df.geometry.total_bounds)
def location_within_bbox(net_bounding_box):
    def f(bus_stop_row):
        location = SUMOCoordinate(x=bus_stop_row.Easting, y=bus_stop_row.Northing)
        return location.to_point().within(net_bounding_box)
    return f       

bus_stops_princess_rd_net_gdf = bus_stops_gdf[bus_stops_gdf.apply(location_within_bbox(net_bounding_box), axis=1)].copy()

def lane_to_edge(lane_id):
    s = lane_id.split('_')
    if len(s) > 2:
        return ''.join(s[:-1])
    return s[0]

bus_to_edge_id = {bs.bus_stop_id.id: lane_to_edge(bs.lane.id) for bs in princess_rd_net.bus_stops}

def all_stops_are_known(bus_stops):
    start = True
    cnt = 0
    for bus_stop in bus_stops:
        if bus_stop in bus_to_edge_id:
            start = False
            cnt += 1
        elif not start and cnt < 2:
            return False
    return cnt > 2

def get_known_stops(bus_stops):
    start = True
    stops = []
    for bus_stop in bus_stops:
        if bus_stop in bus_to_edge_id:
            start = False
            stops.append(bus_stop)
        elif not start and len(stops) < 2:
            return False
    if len(stops) >= 2:
        return stops
    return pd.NA

known_routes_and_stops_gdf = routes_and_stops_gdf[routes_and_stops_gdf['bus_stops'].apply(all_stops_are_known)].copy()
known_routes_and_stops_gdf['bus_stops'] = known_routes_and_stops_gdf['bus_stops'].apply(get_known_stops)
princess_rd_bus_routes_gdf = bus_routes_gdf[bus_routes_gdf['Name'].isin(known_routes_and_stops_gdf['Name'])]
sumo_network = princess_rd_net
sumo_trips = SUMOTrips(sumo_network)
routes = sumo_trips.read_routes()

def make_reachibility_edge_pairs_set(routes):
    reachibility_edge_pairs_set = set()
    route_edges = set()
    for route in tqdm.tqdm(routes, desc='make_reachibility_edge_pairs_set'):
        edges = route['edges'].split(' ')
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                reachibility_edge_pairs_set.add((edges[i], edges[j]))
                route_edges.add(edges[i])
                route_edges.add(edges[j])
    return route_edges, reachibility_edge_pairs_set

route_edges, reachibility_edge_pairs_set = make_reachibility_edge_pairs_set(routes)

def make_graph(route_edges, reachibility_edge_pairs_set):
    reachibility_edge_pairs = list(reachibility_edge_pairs_set)
    G = nx.DiGraph()

    for node in route_edges:
        G.add_node(node)
    for i in range(len(reachibility_edge_pairs)):
        pair = reachibility_edge_pairs[i]
        G.add_edge(pair[0], pair[1])
    return G

G = make_graph(route_edges, reachibility_edge_pairs_set)
R = dict(nx.all_pairs_shortest_path_length(G))

def is_reachable(G, R, edge1, edge2):
    if edge1 not in G.nodes or edge2 not in G.nodes:
        return False
    return R[edge1].get(edge2, 0)

reachible_pairs_count = sum([sum(x.values()) for x in R.values()])
n_edges = len(sumo_network.edges)

def find_longest_good_subsequence(bad_ids, good_ids):
    ids = sorted(list(set(bad_ids + good_ids)))
    bad_ids_set = set(bad_ids)
    result = ''
    for id in ids:
        if id in bad_ids_set:
            if len(result) > 0 and result[-1] != '|':
                result += '|'
        else:
            result += str(id) + ','
    groups = [x.split(',')[:-1] for x in result.split('|')]
    biggest_group = sorted(groups, key=len, reverse=True)[0]
    return list(map(int, biggest_group))

def make_longest_bus_sub_route(route_row):
    bus_stops = route_row['bus_stops']
    edges = [bus_to_edge_id.get(bus_stop, pd.NA) for bus_stop in bus_stops]

    bad_transitions = []
    good_transitions = []
    for i in range(len(edges) - 1):
        edge1 = edges[i]
        edge2 = edges[i + 1]
        if bool(is_reachable(G, R, edge1, edge2)):
            good_transitions.append(i)
        else:
            bad_transitions.append(i)
    longest_good_transitions_seq = find_longest_good_subsequence(bad_transitions, good_transitions)
    new_bus_stops = []
    if longest_good_transitions_seq:
        for index in longest_good_transitions_seq:
            new_bus_stops.append(bus_stops[index])
        last_transition_index = longest_good_transitions_seq[-1]
        new_bus_stops.append(bus_stops[last_transition_index + 1])
    if len(new_bus_stops) >= 2:
        return new_bus_stops
    return pd.NA

def is_bus_route_possible(route_row):
    bus_stops = route_row['bus_stops']
    edge_from = bus_to_edge_id.get(bus_stops[0], pd.NA)
    edge_to   = bus_to_edge_id.get(bus_stops[-1], pd.NA)
    return edge_from == edge_to or bool(is_reachable(G, R, edge_from, edge_to))

def get_bus_stop_edges(bus_stops):
    return [bus_to_edge_id.get(bus_stop, pd.NA) for bus_stop in bus_stops]


routes_and_stops_reachible_gdf = known_routes_and_stops_gdf.copy()
routes_and_stops_reachible_gdf['bus_stops'] = \
routes_and_stops_reachible_gdf.apply(make_longest_bus_sub_route, axis=1)
routes_and_stops_reachible_gdf.dropna(inplace=True)

routes_and_stops_reachible_gdf['from_edge'] = routes_and_stops_reachible_gdf['bus_stops'].apply(lambda x: bus_to_edge_id.get(x[0], pd.NA))
routes_and_stops_reachible_gdf['to_edge'] = routes_and_stops_reachible_gdf['bus_stops'].apply(lambda x: bus_to_edge_id.get(x[-1], pd.NA))

def make_trip(bus_trip_row):
    sumo_bus_trip = SUMOBusTrip(
        bus_trip_id=BusTripId(id=bus_trip_row['Name']),
        depart=0.,
        from_edge=EdgeId(id=bus_trip_row['from_edge']),
        to_edge=EdgeId(id=bus_trip_row['to_edge']),
        duration=20,
        bus_stop_ids=bus_trip_row['bus_stops']
    )
    return sumo_bus_trip

trips = routes_and_stops_reachible_gdf.apply(make_trip, axis=1).to_list()
princess_rd_net.bus_trips = trips
print(len(trips))

princess_rd_net.dir_path = 'gmdata.nosync/princess_rd_pt_net/'
princess_rd_net.name = 'princess_rd_pt_net'
princess_rd_net.enable_bus_trips_and_lanes = False
princess_rd_net.dump_bus_stops()

for edge in princess_rd_net.edges:
    if edge.num_lanes < 2:
        edge.num_lanes = 2

princess_rd_net.save()
princess_rd_net.generate_net_file()


edges_df = princess_rd_net.make_edges_df()
edges_sindex = edges_df.sindex

def find_edge(edges_df, edges_sindex, location: SUMOCoordinate, max_distance=20):
    point = location.to_point()
    indices, distances = edges_sindex.nearest(point, return_distance=True)
    edge_index = indices[1, :][0]
    distance = distances[0]
    if distance > max_distance:
        return None
    return edge_index    


def make_bus_stop(edges_df, edges_sindex, bus_stop_row, max_distance=20, default_lane_num=0):
    location = SUMOCoordinate(x=bus_stop_row.Easting, y=bus_stop_row.Northing)
    edge_index = find_edge(edges_df, edges_sindex, location, max_distance=max_distance)
    if edge_index is None:
        return None
    
    edge = edges_df.iloc[edge_index]
    point = location.to_point()
    edge_position = edge['shape'].project(point)
    startPos = max(0, edge_position - DEFAULT_BUS_STOP_WIDTH / 2)
    endPos = startPos + DEFAULT_BUS_STOP_WIDTH
    
    bus_lane_id = BusLaneId.make(EdgeId(id=edge.edge_id), default_lane_num)

    sumo_bus_stop = SUMOBusStop(
        bus_stop_id=BusStopId(id=bus_stop_row['AtcoCode']),
        name=bus_stop_row['AtcoCode'],
        lane=bus_lane_id,
        startPos=startPos,
        endPos=endPos,
        location=location
    )
    return sumo_bus_stop


bus_stops_list = []
for i in range(len(bus_stops_princess_rd_net_gdf)):
    bus_stop_row = bus_stops_princess_rd_net_gdf.iloc[i]
    bus_stop = make_bus_stop(edges_df, edges_sindex, bus_stop_row, max_distance=10)
    if bus_stop is not None:
        bus_stops_list.append(bus_stop)


princess_rd_net.bus_stops = bus_stops_list
princess_rd_net.dump_bus_stops()
princess_rd_net.save()
princess_rd_net.generate_net_file()
