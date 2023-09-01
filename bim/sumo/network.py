import shapely.geometry.polygon
from shapely.geometry import Point, LineString
from lxml import etree
from typing import List, Dict, Optional
import os
import subprocess
from pydantic import BaseModel
import geopandas as gpd
from shapely import geometry
import pandas as pd
import numpy as np
import tqdm

from ..gis import osm, utils as gis_utils


class NodesXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty nodes element
            self.root = etree.Element("nodes")
            self.tree = etree.ElementTree(self.root)

    def add_node(self, id, x, y, type=None):
        # Create a new node element
        node = etree.SubElement(self.root, "node")
        node.set("id", id)
        node.set("x", str(x))
        node.set("y", str(y))
        if type is not None:
            node.set("type", type)

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True)

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True).decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_nodes(self):
        # Get all nodes as a list of dictionaries
        nodes = []
        for node in self.root.findall("node"):
            nodes.append({
                "id": node.get("id"),
                "x": node.get("x"),
                "y": node.get("y"),
                "type": node.get("type")
            })
        return nodes


class EdgesXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty edges element
            self.root = etree.Element("edges")
            self.tree = etree.ElementTree(self.root)

    def add_edge(self, id: str, from_node: str, to_node: str, priority: int, numLanes: int, speed: float,
                 shape: str = '', bus_lane: Optional[bool] = None):
        # Create a new edge element
        edge = etree.SubElement(self.root, "edge")
        edge.set("id", id)
        edge.set("from", from_node)
        edge.set("to", to_node)
        edge.set("priority", str(priority))
        edge.set("numLanes", str(numLanes))
        edge.set("speed", str(speed))
        edge.set("shape", shape)
        if bus_lane:
            lane = etree.SubElement(edge, "lane")
            lane.set("allow", "bus")
            lane.set("index", "0")

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True)

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True).decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_edges(self):
        # Get all edges as a list of dictionaries
        edges = []
        for edge in self.root.findall("edge"):
            edges.append({
                "id": edge.get("id"),
                "from": edge.get("from"),
                "to": edge.get("to"),
                "priority": edge.get("priority"),
                "numLanes": edge.get("numLanes"),
                "speed": edge.get("speed"),
                "shape": edge.get("shape"),
            })
        return edges


DEFAULT_BUS_VTYPE = "BUS"


class BusStopsXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty edges element
            self.root = etree.Element("additional")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/additional_file.xsd")
            self.tree = etree.ElementTree(self.root)

            # from SUMO documentation: https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html
            vType = etree.SubElement(self.root, "vType")
            vType.set("id", DEFAULT_BUS_VTYPE)
            vType.set("accel", "2.6")
            vType.set("decel", "4.5")
            vType.set("sigma", "0")
            vType.set("length", "10")
            vType.set("minGap", "3")
            vType.set("maxSpeed", "70")
            vType.set("color", "1,1,0")
            vType.set("guiShape", "bus")
            vType.set("vClass", "bus")  # SUMO should add this into their documentation example

    def add_bus_stop(self, id: str, name: str, lane: str, startPos: float, endPos: float, friendlyPos: bool):
        bus_stop = etree.SubElement(self.root, "busStop")
        bus_stop.set("id", id)
        bus_stop.set("name", name)
        bus_stop.set("lane", lane)
        bus_stop.set("startPos", str(startPos))
        bus_stop.set("endPos", str(endPos))
        bus_stop.set("friendlyPos", "true" if friendlyPos else "false")

    def add_bus_trip(self, id: str, depart: float, from_edge: str, to_edge: str, bus_stops_order: List[str],
                     duration: float):
        bus_trip = etree.SubElement(self.root, "trip")
        bus_trip.set("id", id)
        bus_trip.set("type", DEFAULT_BUS_VTYPE)
        bus_trip.set("depart", str(depart))
        bus_trip.set("from", from_edge)
        bus_trip.set("to", to_edge)
        for bus_stop_id in bus_stops_order:
            bus_stop = etree.SubElement(bus_trip, "stop")
            bus_stop.set("busStop", bus_stop_id)
            bus_stop.set("duration", str(duration))

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file


DEFAULT_NODE_TYPE = 'priority'
DEFAULT_EDGE_PRIORITY = 2
DEFAULT_EDGE_MAX_SPEED = 50
DEFAULT_EDGE_NUM_LANES = 2
SUMO_NETWORK_EPSG = gis_utils.EPSG_BNG

USE_BUS_STOPS = True
DEFAULT_BUS_STOP_WIDTH = 10


class NodeId(BaseModel):
    id: str


class EdgeId(BaseModel):
    id: str


class SUMOCoordinate(BaseModel):
    x: float
    y: float
    z: float = 0

    def __str__(self):
        if hasattr(self,'z') and self.z != 0:
            return f"{self.x},{self.y},{self.z}"
        return f"{self.x},{self.y}"

    def to_point(self):
        if hasattr(self,'z') and self.z != 0:
            return Point(self.x, self.y, self.z)
        return Point(self.x, self.y)


class SUMONode(BaseModel):
    node_id: NodeId
    coordinate: SUMOCoordinate
    node_type: str = DEFAULT_NODE_TYPE


class SUMOEdge(BaseModel):
    edge_id: EdgeId
    from_id: NodeId
    to_id: NodeId
    shape: List[SUMOCoordinate] = []
    priority: int = DEFAULT_EDGE_PRIORITY
    num_lanes: int = DEFAULT_EDGE_NUM_LANES
    max_speed: int = DEFAULT_EDGE_MAX_SPEED
    length: float = 0.
    lsoa_code: str = ''
    bus_lane: Optional[bool] = None

    def make_edge_linestring(self):
        return LineString([Point(coord.x, coord.y) for coord in self.shape])


class BusStopId(BaseModel):
    id: str


class BusTripId(BaseModel):
    id: str


class BusLaneId(BaseModel):
    id: str

    @staticmethod
    def make(edge_id: EdgeId, lane_number: int):
        return BusLaneId(id=f'{edge_id.id}_{lane_number}')


class SUMOBusStop(BaseModel):
    bus_stop_id: BusStopId
    name: str
    lane: BusLaneId  # lane with '-' sign means that bus stop is on the opposite side
    startPos: float  # distance from lane start
    endPos: float  # distance from lane start
    location: SUMOCoordinate
    friendlyPos: bool = True


class SUMOBusTrip(BaseModel):
    bus_trip_id: BusStopId
    depart: float
    from_edge: EdgeId
    to_edge: EdgeId
    bus_stop_ids: List[str]
    duration: float


class SUMONetwork(BaseModel, extra='allow'):
    name: str
    dir_path: str
    nodes: List[SUMONode]
    edges: List[SUMOEdge]
    bus_stops: List[SUMOBusStop] = []
    bus_trips: List[SUMOBusTrip] = []
    enable_bus_trips_and_lanes: bool = False

    _node_id_map: Optional[Dict[str, SUMONode]] = None

    @property
    def file_prefix(self):
        return os.path.join(self.dir_path, self.name + '_')

    def make_nodes_path(self):
        return self.file_prefix + 'nodes.nod.xml'

    def make_netconvert_log_path(self):
        return self.file_prefix + 'netconvert.log'

    def dump_nodes(self):
        self.make_dir()
        file_path = self.make_nodes_path()
        nodes_xml = NodesXML()
        for node in self.nodes:
            nodes_xml.add_node(node.node_id.id, node.coordinate.x, node.coordinate.y, node.node_type)
        nodes_xml.save(file_path)

    def make_edges_path(self):
        return self.file_prefix + 'edges.edg.xml'

    def dump_edges(self):
        self.make_dir()
        file_path = self.make_edges_path()
        edges_xml = EdgesXML()
        for edge in self.edges:
            shape_str = ' '.join(list(map(str, edge.shape)))
            edges_xml.add_edge(
                edge.edge_id.id, edge.from_id.id, edge.to_id.id, edge.priority, edge.num_lanes, edge.max_speed,
                shape_str, edge.bus_lane)
        edges_xml.save(file_path)

    def make_bus_stop_path(self):
        return self.file_prefix + 'bus_stops.add.xml'

    def dump_bus_stops(self):
        self.make_dir()
        file_path = self.make_bus_stop_path()
        bus_stops_xml = BusStopsXML()
        for bus_stops in self.bus_stops:
            bus_stops_xml.add_bus_stop(
                bus_stops.bus_stop_id.id, bus_stops.name, bus_stops.lane.id, bus_stops.startPos, bus_stops.endPos,
                bus_stops.friendlyPos)

        if self.enable_bus_trips_and_lanes:
            for bus_trip in self.bus_trips:
                bus_stops_xml.add_bus_trip(
                    bus_trip.bus_trip_id.id, bus_trip.depart, bus_trip.from_edge.id, bus_trip.to_edge.id,
                    bus_trip.bus_stop_ids, bus_trip.duration)
        bus_stops_xml.save(file_path)

    def make_net_path(self):
        return self.file_prefix + 'net.net.xml'

    def run_netconvert(self):
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
        netconvert_log_path = self.make_netconvert_log_path()
        node_path = self.make_nodes_path()
        edge_path = self.make_edges_path()
        net_path = self.make_net_path()
        cmd = [
            "netconvert",
            "--lefthand",
            "--node-files={}".format(node_path),
            "--edge-files={}".format(edge_path),
            "--output-file={}".format(net_path)
        ]

        with open(netconvert_log_path, "wb") as f:
            subprocess.run(cmd, stdout=f, stderr=f)

    def generate_net_file(self, use_bus_stops: bool = USE_BUS_STOPS):
        self.dump_nodes()
        self.dump_edges()
        if use_bus_stops:
            self.dump_bus_stops()
        self.run_netconvert()

    def get_node(self, node_id: NodeId):
        if self._node_id_map is None:
            self._node_id_map = {}
            for node in self.nodes:
                self._node_id_map[node.node_id.id] = node
        return self._node_id_map.get(node_id.id)

    def make_dir(self):
        os.makedirs(self.dir_path, exist_ok=True)

    def make_json_path(self):
        return self.make_json_path_from_dir_and_name(self.dir_path, self.name)

    @staticmethod
    def make_json_path_from_dir_and_name(dir_path, network_name):
        return os.path.join(dir_path, f'{network_name}.json')

    def save(self):
        self.make_dir()
        with open(self.make_json_path(), 'w+') as file:
            file.write(self.json())

    @staticmethod
    def try_load(dir_path: str, network_name: str):
        json_path = SUMONetwork.make_json_path_from_dir_and_name(dir_path, network_name)
        if os.path.exists(json_path):
            network = SUMONetwork.parse_file(json_path)
            return network
        return None

    def make_streets_gdf(self):
        """
        geopandas geometry here - coordinates of the edge.from_node
        """
        node_coordinates = []
        self._node_id_map = None  # better to remove the kludge
        for i in range(len(self.edges)):
            edge = self.edges[i]
            from_node = self.get_node(edge.from_id)
            row = (edge.edge_id.id, Point(from_node.coordinate.x, from_node.coordinate.y), edge.length, edge.lsoa_code)
            node_coordinates.append(row)

        streets_gdf = gpd.GeoDataFrame(
            node_coordinates,
            columns=['edge_id', 'point', 'street_length', gis_utils.LSOA_CODE_COLUMN],
            geometry=[row[1] for row in node_coordinates]
        )
        streets_gdf.crs = gis_utils.EPSG_BNG
        return streets_gdf.to_crs(gis_utils.EPSG_WSG84)

    def make_edges_df(self):
        """
        geopandas geometry here - LineString converted from edge.shape
        """
        edges_df = gpd.GeoDataFrame(
            [{
                'edge_id': edge.edge_id.id,
                'from_id': edge.from_id.id,
                'to_id': edge.to_id.id,
                'shape': edge.make_edge_linestring(),
                'priority': edge.priority,
                'num_lanes': edge.num_lanes,
                'max_speed': edge.max_speed,
                'length': edge.length,
                'lsoa_code': edge.lsoa_code,
            } for edge in self.edges],
            geometry='shape'
        )
        edges_df.crs = gis_utils.EPSG_BNG
        return edges_df

    def make_nodes_df(self):
        nodes_df = gpd.GeoDataFrame(
            [{
                'node_id': node.node_id.id,
                'point': node.coordinate.to_point(),
                'type': node.node_type,
            } for node in self.nodes],
            geometry='point'
        )
        nodes_df.crs = gis_utils.EPSG_BNG
        return nodes_df


def __old_extract_sumo_nodes_from_streets_gdf(streets_gdf, max_nodes_dist: Optional[float] = None):
    nodes_gdf = gpd.GeoDataFrame([])
    if max_nodes_dist is None:
        max_nodes_dist = 0.1

    def get_closest_nodes(nodes_df, max_distance):
        def f(node_point):
            nodes_spatial_index = nodes_df.sindex
            possible_matches_index = list(nodes_spatial_index.nearest(node_point))
            possible_matches_index = np.asarray(possible_matches_index)[1, :]
            possible_matches_index = list(set(possible_matches_index))
            possible_matches = nodes_df.iloc[possible_matches_index]
            return possible_matches[(possible_matches.distance(node_point) <= max_distance)]

        return f

    def get_first_and_last_coords(geometry):
        nonlocal nodes_gdf

        coords = list(geometry.coords)
        first_and_last = [(coords[0], 0)] + [(coords[-1], -1)] if len(coords) > 1 else []
        result = []
        new_geometry = geometry
        for coord, coord_position in first_and_last:
            # coord_position == 0 means first coord, coord_position == -1 means last coord
            node_id = None
            if len(nodes_gdf):
                get_closest_nodes_f = get_closest_nodes(nodes_gdf, max_nodes_dist)
                found_closest_points_df = get_closest_nodes_f(Point(*coord))
                if len(found_closest_points_df) > 0:
                    closest_point = found_closest_points_df.iloc[0]
                    node_id = closest_point['node_id']
                    new_geometry = gis_utils.replace_linestring_coordinate(new_geometry, coord_position, closest_point.geometry)

            if node_id is None:
                new_node_id = len(nodes_gdf)
                new_node_df = gpd.GeoDataFrame([{'node_id': new_node_id}], geometry=gpd.points_from_xy([coord[0]], [coord[1]]))
                nodes_gdf = pd.concat([nodes_gdf, new_node_df])
                node_id = new_node_id

            result.append(node_id)
        return result, new_geometry

    streets_gdf['nodes'] = None
    # streets_gdf['nodes'] = streets_gdf['geometry'].apply(get_first_and_last_coords)
    for i in range(len(streets_gdf)):
        street = streets_gdf.iloc[i]
        street_index = street.name
        street_nodes, street_geometry = get_first_and_last_coords(street.geometry)
        streets_gdf.at[street_index, 'nodes'] = street_nodes
        streets_gdf.at[street_index, 'geometry'] = street_geometry

    # nodes = []
    # for node_id, geometry in zip(nodes_gdf['node_id'], nodes_gdf['geometry']):
    #     coords = geometry.coords[0]  # Points has one coordinate pair
    #     coord = SUMOCoordinate(x=coords[0], y=coords[1])
    #     node = SUMONode(node_id=NodeId(id=f'node{node_id}'), coordinate=coord)
    #     nodes.append(node)

    nodes = extract_sumo_nodes_from_nodes_gdf(nodes_gdf)
    return nodes


def extract_sumo_nodes_from_nodes_gdf(nodes_gdf):
    nodes = []
    for _, row in nodes_gdf.iterrows():
        node_id = row.node_id
        node_type = DEFAULT_NODE_TYPE
        if 'node_type' in row:
            node_type = row.node_type

        if 'point' in row:
            coords = row.point.coords[0]
            coord = SUMOCoordinate(x=coords[0], y=coords[1])
            node = SUMONode(node_id=NodeId(id=node_id), coordinate=coord, node_type=node_type)
        else:
            coords = row.geometry.coords[0]
            coord = SUMOCoordinate(x=coords[0], y=coords[1])
            node = SUMONode(node_id=NodeId(id=f'node{node_id}'), coordinate=coord, node_type=node_type)
        nodes.append(node)
    return nodes


def extract_sumo_nodes_and_update_edges_from_streets_gdf(streets_gdf, max_nodes_dist: Optional[float] = None):
    nodes_gdf = gpd.GeoDataFrame([])
    max_node_to_edge_dist = 2  # meters
    if max_nodes_dist is None:
        max_nodes_dist = 0.1

    def get_closest_nodes(nodes_df, max_distance):
        def f(node_point):
            nodes_spatial_index = nodes_df.sindex
            possible_matches_index = list(nodes_spatial_index.nearest(node_point))
            possible_matches_index = np.asarray(possible_matches_index)[1, :]
            possible_matches_index = list(set(possible_matches_index))
            possible_matches = nodes_df.iloc[possible_matches_index]
            return possible_matches[(possible_matches.distance(node_point) <= max_distance)]

        return f

    def get_closest_geometry(gdf, max_distance):
        def f(geometry):
            gdf_spatial_index = gdf.sindex
            possible_matches_index = list(gdf_spatial_index.nearest(geometry))
            possible_matches_index = np.asarray(possible_matches_index)[1, :]
            possible_matches_index = list(set(possible_matches_index))
            possible_matches = gdf.iloc[possible_matches_index]
            return possible_matches[(possible_matches.distance(geometry) <= max_distance)]

        return f

    def get_first_and_last_coords(geometry):
        nonlocal nodes_gdf

        coords = list(geometry.coords)
        first_and_last = [(coords[0], 0)] + [(coords[-1], -1)] if len(coords) > 1 else []
        result = []
        new_geometry = geometry
        for coord, coord_position in first_and_last:
            # coord_position == 0 means first coord, coord_position == -1 means last coord
            node_id = None
            if len(nodes_gdf):
                get_closest_nodes_f = get_closest_nodes(nodes_gdf, max_nodes_dist)
                found_closest_points_df = get_closest_nodes_f(Point(*coord))
                if len(found_closest_points_df) > 0:
                    closest_point = found_closest_points_df.iloc[0]
                    node_id = closest_point['node_id']
                    new_geometry = gis_utils.replace_linestring_coordinate(new_geometry, coord_position, closest_point.geometry)

            if node_id is None:
                new_node_id = len(nodes_gdf)
                new_node_df = gpd.GeoDataFrame([{'node_id': new_node_id}], geometry=gpd.points_from_xy([coord[0]], [coord[1]]))
                nodes_gdf = pd.concat([nodes_gdf, new_node_df])
                node_id = new_node_id

            result.append(node_id)
        if result[0] == result[-1]:
            # delete bad edges, like loops (end node == start node)
            # it is necessary to avoid making trips for bad edges, because SUMO's netconvert will delete it
            return None, None
        return result, new_geometry

    def split_edge(edges_df, edge_index, point_position_on_edge, split_node_int):
        edge = edges_df.loc[edge_index]
        geometry_split = gis_utils.split_linestring(edge['geometry'], point_position_on_edge)
        if len(geometry_split) != 2 or edge['nodes'][0] == split_node_int or split_node_int == edge['nodes'][1]:
            return None, None

        edge1 = edge.copy()
        edge1['id'] = str(edge1['id']) + '_1'
        edge1['name'] = str(edge1['name']) + '_1'
        edge1['coords'] = ''
        edge1['geometry'] = geometry_split[0]
        edge1['nodes'] = [edge['nodes'][0], split_node_int]

        edge2 = edge.copy()
        edge2['id'] = str(edge1['id']) + '_2'
        edge2['name'] = str(edge2['name']) + '_2'
        edge2['coords'] = ''
        edge2['geometry'] = geometry_split[1]
        edge2['nodes'] = [split_node_int, edge['nodes'][1]]

        return edge1, edge2

    def process_node_to_split_edges(node_index):
        nonlocal nodes_gdf, streets_gdf

        eps = 0.01  # 1 percent of edge length
        node = nodes_gdf.iloc[node_index]
        point = node.geometry
        node_id = node.node_id
        # we change edges, so we have to use latest updated streets_gdf each time, we cannot cache function
        get_closest_edges_f = get_closest_geometry(streets_gdf, max_node_to_edge_dist)
        found_closest_edges_df = get_closest_edges_f(point)
        for i in range(len(found_closest_edges_df)):
            edge = found_closest_edges_df.iloc[i]
            point_position_on_edge = edge.geometry.project(point, normalized=True)
            if eps < point_position_on_edge < 1 - eps:
                edge_index = edge.name
                edge1, edge2 = split_edge(streets_gdf, edge_index, point_position_on_edge, node_id)
                if edge1 is None:
                    pass  # something wrong with split, do nothing here
                else:
                    # replace old edge with first split
                    streets_gdf.loc[edge_index] = edge1
                    # add edge to the end of dataframe
                    edge2_df = gpd.GeoDataFrame([edge2])
                    edge2_df.crs = streets_gdf.crs
                    streets_gdf = pd.concat([streets_gdf, edge2_df], ignore_index=True)


    streets_gdf['nodes'] = None
    # streets_gdf['nodes'] = streets_gdf['geometry'].apply(get_first_and_last_coords)
    streets_to_delete = []
    for i in tqdm.tqdm(range(len(streets_gdf)), desc='process_nodes'):
        street = streets_gdf.iloc[i]
        street_index = street.name
        street_nodes, street_geometry = get_first_and_last_coords(street.geometry)
        if street_nodes is None:
            streets_to_delete.append(street_index)  # delete bad edges, like loops (end node == start node)
        else:
            streets_gdf.at[street_index, 'nodes'] = street_nodes
            streets_gdf.at[street_index, 'geometry'] = street_geometry
    if len(streets_to_delete) > 0:
        streets_gdf.drop(streets_to_delete, inplace=True)
        streets_gdf.reset_index(drop=True, inplace=True)

    # process edges in reverse order - to safely delete divided edges
    n = len(streets_gdf)
    for i in tqdm.tqdm(range(n - 1, -1, -1), desc='split_edges'):
        edge = streets_gdf.iloc[i]
        for node_index in edge.nodes:
            process_node_to_split_edges(node_index)

    sumo_nodes = []
    for node_id, geometry in tqdm.tqdm(zip(nodes_gdf['node_id'], nodes_gdf['geometry']), desc='make_sumo_nodes'):
        coords = geometry.coords[0]  # Points has one coordinate pair
        coord = SUMOCoordinate(x=coords[0], y=coords[1])
        sumo_node = SUMONode(node_id=NodeId(id=f'node{node_id}'), coordinate=coord)
        sumo_nodes.append(sumo_node)

    return sumo_nodes, streets_gdf


def extract_sumo_edges_from_edge_gdf(edges_df, num_lanes=DEFAULT_EDGE_NUM_LANES):
    if 'nodes' not in edges_df and 'from_id' not in edges_df:
        raise Exception("make 'nodes' column in edges_df: call extract_sumo_nodes_and_update_edges_from_streets_gdf()")

    edges = []
    default_num_lanes = num_lanes
    for i in range(len(edges_df)):
        row = edges_df.iloc[i]
        if 'nodes' in row:
            edge_id = f'edge{i}'
            first_and_last_nodes = row['nodes']
            node_from = f'node{first_and_last_nodes[0]}'
            node_to = f'node{first_and_last_nodes[1]}'
            row_geometry = row.geometry
            lsoa_code = row['LSOA_code']
        else:
            edge_id = row['edge_id']
            node_from = row['from_id']
            node_to = row['to_id']
            row_geometry = row['shape']
            lsoa_code = row['lsoa_code']
        shape = [SUMOCoordinate(x=c[0], y=c[1]) for c in row_geometry.coords]
        max_speed = DEFAULT_EDGE_MAX_SPEED
        if 'max_speed' in row:
            max_speed = row['max_speed']
        num_lanes = default_num_lanes
        if 'num_lanes' in row:
            num_lanes = row['num_lanes']
        edge = SUMOEdge(
            edge_id=EdgeId(id=edge_id),
            from_id=NodeId(id=node_from),
            to_id=NodeId(id=node_to),
            shape=shape,
            length=row_geometry.length,
            lsoa_code=lsoa_code,
            num_lanes=num_lanes,
            max_speed=max_speed
        )
        edges.append(edge)
    return edges


def extract_sumo_bus_stops_from_bus_stops_gdf(edges_df, num_lanes=DEFAULT_EDGE_NUM_LANES):
    edges = []
    for i in range(len(edges_df)):
        row = edges_df.iloc[i]
        first_and_last_nodes = row.nodes
        node_from = f'node{first_and_last_nodes[0]}'
        node_to = f'node{first_and_last_nodes[1]}'
        shape = [SUMOCoordinate(x=c[0], y=c[1]) for c in row.geometry.coords]
        edge = SUMOEdge(
            edge_id=EdgeId(id=f'edge{i}'),
            from_id=NodeId(id=node_from),
            to_id=NodeId(id=node_to),
            shape=shape,
            length=row.geometry.length,
            lsoa_code=row.LSOA_code,
            num_lanes=num_lanes
        )
        edges.append(edge)
    return edges


def extract_bus_stops(edges_df):
    bus_stops_gdf = gpd.read_file('gmdata.nosync/TfGMStoppingPoints.csv')
    bus_stops_gdf = bus_stops_gdf[bus_stops_gdf['Status'] == 'act']
    bus_top_locations = gpd.points_from_xy(bus_stops_gdf['Easting'], bus_stops_gdf['Northing'])
    bus_stops_gdf.set_geometry(bus_top_locations, inplace=True)
    bus_stops_gdf.crs = 27700

    def find_edge(edges_sindex, location: SUMOCoordinate, max_distance=20):
        point = location.to_point()
        indices, distances = edges_sindex.nearest(point, return_distance=True)
        edge_index = indices[1, :][0]
        distance = distances[0]
        if distance > max_distance:
            return None
        return edge_index

    def make_bus_stop(edges_df, edges_sindex, bus_stop_row, max_distance=20, default_lane_num=0):
        location = SUMOCoordinate(x=bus_stop_row.Easting, y=bus_stop_row.Northing)
        edge_index = find_edge(edges_sindex, location, max_distance=max_distance)
        if edge_index is None:
            # atco_code = bus_stop_row['AtcoCode']
            # print(f'drop bus stop {atco_code}: too distant from edges')
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

    net_bounding_box = geometry.box(*edges_df.geometry.total_bounds)

    def location_within_bbox(net_bounding_box):
        def f(bus_stop_row):
            location = SUMOCoordinate(x=bus_stop_row.Easting, y=bus_stop_row.Northing)
            return location.to_point().within(net_bounding_box)

        return f

    edges_sindex = edges_df.sindex
    bus_stops_stockport_gdf = bus_stops_gdf[bus_stops_gdf.apply(location_within_bbox(net_bounding_box), axis=1)].copy()
    bus_stops_list = []
    for i in range(len(bus_stops_stockport_gdf)):
        bus_stop_row = bus_stops_stockport_gdf.iloc[i]
        bus_stop = make_bus_stop(edges_df, edges_sindex, bus_stop_row, max_distance=10)
        if bus_stop is not None:
            bus_stops_list.append(bus_stop)
    return bus_stops_list


def make_gm_sub_net(
        network_name: str,
        network_dir_path: str,
        boundary_polygon: Optional[shapely.geometry.polygon.Polygon] = None,
        num_lanes: int = DEFAULT_EDGE_NUM_LANES,
        max_nodes_dist: Optional[float] = None,
        gdf_path: str = osm.OLD_STREETS_GDF_PATH
):
    """
        1) get all streets from nominatim
        2) extract streets in the specified rectangle area (boundary_polygon)
        3) street begin and ends - nodes location
        4) street coordinates turns into edge parameter: shape="5292.46,2928.69 5242.79,2932.27 ..."
        5) save it to json via sumo_network.json()
    """
    print(f'start creating sumo net "{network_name}"')
    streets_gdf = osm.Nominatim().load_streets_gdf(gdf_path=gdf_path)

    if boundary_polygon:
        streets_gdf = streets_gdf[streets_gdf.geometry.within(boundary_polygon)].copy()
    streets_gdf = gis_utils.convert_epsg(streets_gdf, SUMO_NETWORK_EPSG)  # to work in meters

    print(f'streets_gdf loaded with {len(streets_gdf)} rows')

    sumo_nodes, edges_df = extract_sumo_nodes_and_update_edges_from_streets_gdf(streets_gdf, max_nodes_dist)
    sumo_edges = extract_sumo_edges_from_edge_gdf(edges_df, num_lanes)

    bus_stops_list = extract_bus_stops(edges_df)

    sumo_network = SUMONetwork(name=network_name, dir_path=network_dir_path, nodes=sumo_nodes, edges=sumo_edges,
                               bus_stops=bus_stops_list)
    sumo_network.save()
    sumo_network.generate_net_file()

    return sumo_network


def get_stockport_center_boundary():
    from shapely.geometry.polygon import Polygon
    min_lat = 53.40493052689684
    max_lat = 53.41636933551804
    min_lon = -2.1430687886598005
    max_lon = -2.178263457725206
    return Polygon.from_bounds(min_lon, min_lat, max_lon, max_lat)


def get_princess_rd_a6_boundary():
    from shapely.geometry.polygon import Polygon
    min_y = 391000
    max_y = 396500
    min_x = 381400
    max_x = 387800
    return Polygon.from_bounds(min_x, min_y, max_x, max_y)


DEFAULT_MAX_NODE_MERGE_DISTANCE = 10


def make_base_stockport_net():
    return make_gm_sub_net(
        'stockport_base_model',
        'gmdata.nosync/stockport_base_model/',
        get_stockport_center_boundary()
    )


def make_fixed_stockport_net(max_nodes_dist=DEFAULT_MAX_NODE_MERGE_DISTANCE):
    return make_gm_sub_net(
        'stockport_fixed_base_model',
        'gmdata.nosync/stockport_fixed_base_model/',
        get_stockport_center_boundary(),
        max_nodes_dist=max_nodes_dist,
        gdf_path=osm.FIXED_STREETS_GDF_PATH
    )


def make_base_gm_net():
    return make_gm_sub_net(
        'gm_base_model',
        'gmdata.nosync/gm_base_model/'
    )


def make_not_final_gm_net(max_nodes_dist=DEFAULT_MAX_NODE_MERGE_DISTANCE):
    return make_gm_sub_net(
        'gm_fixed_base_model',
        'gmdata.nosync/gm_fixed_base_model/',
        max_nodes_dist=max_nodes_dist,
        gdf_path=osm.OLD_STREETS_GDF_PATH
    )


def make_fixed_gm_net(max_nodes_dist=DEFAULT_MAX_NODE_MERGE_DISTANCE):
    return make_gm_sub_net(
        'gm_osm_fixed_nl_ms_base_model',
        'gmdata.nosync/gm_osm_fixed_nl_ms_base_model/',
        max_nodes_dist=max_nodes_dist,
        gdf_path=osm.FIXED_STREETS_GDF_PATH
    )


if __name__ == '__main__':
    make_fixed_gm_net()
