from .network import SUMONetwork, get_princess_rd_a6_boundary, extract_sumo_edges_from_edge_gdf, \
    extract_sumo_nodes_from_nodes_gdf
from ..traffic.trips import SUMOTrips, TripsXML
import matplotlib.pyplot as plt
import pandas as pd
import os


def extract_routes_within_boundary(edges_index):
    def f(edges_str):
        edges_list = edges_str.split(' ')
        routes = []
        route = None
        for edge in edges_list + [None]:
            if edge in edges_index:
                if route is None:
                    route = [edge]
                else:
                    route.append(edge)
            elif route is not None:
                routes.append(route)
                route = None
        return routes

    return f


def plot_splitted_origin_route(route, extract_routes_f, sub_area_edges_gdf, edges_gdf):
    extracted_routes = extract_routes_f(route['edges'])
    fig, ax = plt.subplots()
    sub_area_edges_gdf.plot(ax=ax)
    origin_route = route['edges'].split(' ')
    edges_gdf.loc[origin_route].plot(ax=ax, color='yellow').plot(ax=ax)
    for i in range(len(extracted_routes)):
        sub_area_edges_gdf.loc[extracted_routes[i]].plot(ax=ax, color='red')
    return extracted_routes


class TrafficNet:
    def __init__(self, sumo_net: SUMONetwork):
        self.sumo_net = sumo_net

    def make_sub_traffic_net(self, boundary_polygon, new_sub_traffic_net_dir: str, new_sub_traffic_net_name: str,
                             verbose: bool = False):
        new_sumo_network, sub_area_edges_gdf = self.make_sub_net(boundary_polygon, new_sub_traffic_net_dir,
                                                                 new_sub_traffic_net_name, verbose)
        self.make_sub_routes(new_sumo_network, sub_area_edges_gdf)
        return TrafficNet(new_sumo_network)

    def make_sub_net(self, boundary_polygon, new_sub_traffic_net_dir: str, new_sub_traffic_net_name: str,
                     verbose: bool = False):
        net_path = SUMONetwork(name=new_sub_traffic_net_name, dir_path=new_sub_traffic_net_dir,
                               nodes=[], edges=[]).make_json_path()
        if os.path.exists(net_path):
            new_sumo_network = SUMONetwork.parse_file(net_path)
            return new_sumo_network, new_sumo_network.make_edges_df()
        edges_gdf = self.sumo_net.make_edges_df()
        nodes_gdf = self.sumo_net.make_nodes_df()
        sub_area_edges_gdf = edges_gdf[edges_gdf.geometry.intersects(boundary_polygon)].copy()
        sub_area_sumo_edges = extract_sumo_edges_from_edge_gdf(sub_area_edges_gdf)

        node_ids = set(sub_area_edges_gdf['from_id']).union(set(sub_area_edges_gdf['to_id']))
        nodes_gdf.set_index('node_id', inplace=True)
        sub_area_nodes_gdf = nodes_gdf.loc[node_ids].copy().reset_index()
        sub_area_sumo_nodes = extract_sumo_nodes_from_nodes_gdf(sub_area_nodes_gdf)

        bus_stops_list = []
        for bus_stop in self.sumo_net.bus_stops:
            if bus_stop.location.to_point().intersects(boundary_polygon):
                bus_stops_list.append(bus_stop)

        if verbose:
            print(f'edges ratio: {len(sub_area_edges_gdf) / len(edges_gdf)}')
            print(f'nodes ratio: {len(sub_area_nodes_gdf) / len(nodes_gdf)}')
            print(f'bus stops ratio: {len(bus_stops_list) / len(self.sumo_net.bus_stops)}')

        new_sumo_network = SUMONetwork(name=new_sub_traffic_net_name, dir_path=new_sub_traffic_net_dir,
                                       nodes=sub_area_sumo_nodes, edges=sub_area_sumo_edges, bus_stops=bus_stops_list)
        os.makedirs(new_sub_traffic_net_dir, exist_ok=True)
        new_sumo_network.save()
        new_sumo_network.generate_net_file()
        return new_sumo_network, sub_area_edges_gdf

    def make_sub_routes(self, new_sumo_network, sub_area_edges_gdf):
        sumo_trips = SUMOTrips(self.sumo_net)
        routes = sumo_trips.read_routes()
        sub_area_edges_gdf.set_index('edge_id', inplace=True)
        extract_routes = extract_routes_within_boundary(sub_area_edges_gdf.index)
        routes_df = pd.DataFrame(routes)

        extracted_routes_df = routes_df.copy()
        extracted_routes_df['extracted'] = extracted_routes_df['edges'].apply(extract_routes)

        extracted_routes_df = extracted_routes_df.explode('extracted').dropna()
        del extracted_routes_df['edges']
        extracted_routes_df['extracted'] = extracted_routes_df['extracted'].apply(' '.join)
        extracted_routes_df = extracted_routes_df.rename({'extracted': 'edges'}, axis=1).reset_index(drop=True)
        extracted_routes_df['id'] = extracted_routes_df['id'] + \
                                    pd.Series(extracted_routes_df.index).apply(lambda x: '_' + str(x))

        new_sumo_trips = SUMOTrips(new_sumo_network)
        new_sumo_trips.update_routes(extracted_routes_df)
        new_sumo_trips.update_trips(extracted_routes_df)


    @staticmethod
    def make_default():
        """
        Making full GM Traffic net from scratch:
        1) run python3 network.py (or import bim.sumo.network and run make_fixed_gm_net())
        2) run RegionExperiment with default configuration
        """
        net_path = 'gmdata.nosync/whole_gm_experiments_new_ml_ms/run0/gm_run0.json'
        gm_osm_fixed_nl_ms_net = SUMONetwork.parse_file(net_path)
        return TrafficNet(gm_osm_fixed_nl_ms_net)

    def make_princess_rd_to_a6_subnet(self, verbose: bool = False):
        dir_path = 'gmdata.nosync/princess_rd_base_net'
        network_name = 'princess_rd_base_net'
        sumo_network = SUMONetwork.try_load(dir_path, network_name)
        if sumo_network:
            return TrafficNet(sumo_network)
        boundary_polygon = get_princess_rd_a6_boundary()
        return self.make_sub_traffic_net(boundary_polygon, dir_path, network_name, verbose)
