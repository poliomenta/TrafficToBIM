import shapely.geometry.polygon
from lxml import etree
from typing import List, Dict, Optional
import os
import subprocess
from pydantic import BaseModel

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

    def add_edge(self, id: str, from_node: str, to_node: str, priority: int, numLanes: int, speed: float, shape: str = ''):
        # Create a new edge element
        edge = etree.SubElement(self.root, "edge")
        edge.set("id", id)
        edge.set("from", from_node)
        edge.set("to", to_node)
        edge.set("priority", str(priority))
        edge.set("numLanes", str(numLanes))
        edge.set("speed", str(speed))
        edge.set("shape", shape)

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



DEFAULT_NODE_TYPE = 'priority'
DEFAULT_EDGE_PRIORITY = 2
DEFAULT_EDGE_MAX_SPEED = 50
DEFAULT_EDGE_NUM_LANES = 2
SUMO_NETWORK_EPSG = gis_utils.EPSG_BNG


class NodeId(BaseModel):
    id: str


class EdgeId(BaseModel):
    id: str


class SUMOCoordinate(BaseModel):
    x: float
    y: float

    def __str__(self):
        return f"{self.x},{self.y}"


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


class SUMONetwork(BaseModel, extra='allow'):
    name: str
    dir_path: str
    nodes: List[SUMONode]
    edges: List[SUMOEdge]

    _node_id_map: Optional[Dict[str, SUMONode]] = None

    @property
    def file_prefix(self):
        return os.path.join(self.dir_path, self.name + '_')

    def make_nodes_path(self):
        return self.file_prefix + 'nodes.nod.xml'

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
                edge.edge_id.id, edge.from_id.id, edge.to_id.id, edge.priority, edge.num_lanes, edge.max_speed, shape_str)
        edges_xml.save(file_path)

    def make_net_path(self):
        return self.file_prefix + 'net.net.xml'

    def run_netconvert(self):
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
        node_path = self.make_nodes_path()
        edge_path = self.make_edges_path()
        net_path = self.make_net_path()
        cmd = [
            "netconvert",
            "--node-files={}".format(node_path),
            "--edge-files={}".format(edge_path),
            "--output-file={}".format(net_path)
        ]

        subprocess.run(cmd)

    def generate_net_file(self):
        self.dump_nodes()
        self.dump_edges()
        self.run_netconvert()

    def get_node(self, node_id: NodeId):
        if self._node_id_map is None:
            self._node_id_map = {}
            for node in self.nodes:
                self._node_id_map[node.node_id.id] = node
        return self._node_id_map.get(node_id.id)

    def make_dir(self):
        os.makedirs(self.dir_path, exist_ok=True)

    def save(self):
        self.make_dir()
        with open(os.path.join(self.dir_path, f'{self.name}.json'), 'w+') as file:
            file.write(self.json())

    def make_streets_gdf(self):
        from shapely.geometry import Point
        import geopandas as gpd
        node_coordinates = []
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


def extract_sumo_nodes_from_streets_gdf(streets_gdf):
    coords_to_node = {}

    def get_first_and_last_coords(geometry):
        coords = list(geometry.coords)
        first_and_last = [coords[0]] + [coords[-1]] if len(coords) > 1 else []
        result = []
        for coords in first_and_last:
            if coords not in coords_to_node:
                coords_to_node[coords] = len(coords_to_node)
            result.append(coords_to_node[coords])
        return result

    streets_gdf['nodes'] = streets_gdf['geometry'].apply(get_first_and_last_coords)

    nodes = []
    for coords, node_id in coords_to_node.items():
        coord = SUMOCoordinate(x=coords[0], y=coords[1])
        node = SUMONode(node_id=NodeId(id=f'node{node_id}'), coordinate=coord)
        nodes.append(node)

    return nodes


def extract_sumo_edges_from_streets_gdf(streets_gdf, num_lanes=DEFAULT_EDGE_NUM_LANES):
    if not 'nodes' in streets_gdf:
        raise Exception("make 'nodes' column in streets_gdf, call extract_sumo_nodes_from_streets_gdf()")

    edges = []
    for i in range(len(streets_gdf)):
        row = streets_gdf.iloc[i]
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


def make_gm_sub_net(
        network_name: str,
        network_dir_path: str,
        boundary_polygon: Optional[shapely.geometry.polygon.Polygon] = None,
        num_lanes: int = DEFAULT_EDGE_NUM_LANES,
    ):
    """
        1) get all streets from nominatim
        2) extract streets in the specified rectangle area (boundary_polygon)
        3) street begin and ends - nodes location
        4) street coordinates turns into edge parameter: shape="5292.46,2928.69 5242.79,2932.27 ..."
        5) save it to json via sumo_network.json()
    """
    streets_gdf = osm.Nominatim().load_streets_gdf()

    if boundary_polygon:
        streets_gdf = streets_gdf[streets_gdf.geometry.within(boundary_polygon)].copy()
    streets_gdf = gis_utils.convert_epsg(streets_gdf, SUMO_NETWORK_EPSG)

    nodes = extract_sumo_nodes_from_streets_gdf(streets_gdf)
    edges = extract_sumo_edges_from_streets_gdf(streets_gdf, num_lanes)

    sumo_network = SUMONetwork(name=network_name, dir_path=network_dir_path, nodes=nodes, edges=edges)
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


def make_base_stockport_net():
    return make_gm_sub_net(
        'stockport_base_model',
        'gmdata.nosync/stockport_base_model/',
        get_stockport_center_boundary()
    )


def make_base_gm_net():
    return make_gm_sub_net(
        'gm_base_model',
        'gmdata.nosync/gm_base_model/'
    )


if __name__ == '__main__':
    make_base_gm_net()
