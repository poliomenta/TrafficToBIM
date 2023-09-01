#!/usr/bin/env python
# coding: utf-8

"""
1) Update network:
1.1) Get edges, nodes, and bus stops which intersects the boundary area
1.2) Generate SUMOEdge, SUMONode, SUMOBusStop from gdfs
1.3) Put back in sumo_network
1.4) Save to new folder

2) Update routes:
2.1) Read routes from GM base net and select all routes with edges from step 1
2.2) for each route make list of its edges, then make sub-routes by splitting by removed edges
 -> add splitter routes to the final routes list
"""
from bim.sumo.traffic_net import TrafficNet, extract_routes_within_boundary, plot_splitted_origin_route
from bim.traffic.trips import SUMOTrips, TripsXML

traffic_net = TrafficNet.make_default()
princess_rd_to_a6_subnet = traffic_net.make_princess_rd_to_a6_subnet(True)
princess_rd_to_a6_subnet.sumo_net.make_json_path()
sumo_trips = SUMOTrips(princess_rd_to_a6_subnet.sumo_net)
routes = sumo_trips.read_routes()
sub_area_edges_gdf = princess_rd_to_a6_subnet.sumo_net.make_edges_df()
sub_area_edges_gdf.set_index('edge_id', inplace=True)
extract_routes_f = extract_routes_within_boundary(sub_area_edges_gdf.index)
edges_gdf = traffic_net.sumo_net.make_edges_df()
edges_gdf.set_index('edge_id', inplace=True)
sumo_original_trips = SUMOTrips(traffic_net.sumo_net)
original_routes = sumo_original_trips.read_routes()
extracted_routes = plot_splitted_origin_route(original_routes[874], extract_routes_f, sub_area_edges_gdf, edges_gdf)

# ### Make trips from routes file
trips_xml = TripsXML()
n = 0
for route in routes:
    edges = route['edges'].split(' ')
    if edges[0] == edges[-1]:
        continue
    trips_xml.add_trip(route['id'], route['depart'], edges[0], edges[-1])
    n += 1
trips_xml.save(sumo_trips.trips_path)
