from ..gis.traffic.od_matrix import AreaOriginDestinationMatrix
from ..sumo.network import SUMONetwork
from ..gis import utils as gis_utils
from typing import Optional
import numpy as np
import subprocess
import os
from lxml import etree


class TripsXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty routes element
            self.root = etree.Element("routes")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/routes_file.xsd")
            self.tree = etree.ElementTree(self.root)

    def add_trip(self, id, depart, from_edge, to_edge):
        # Create a new trip element
        trip = etree.SubElement(self.root, "trip")
        trip.set("id", id)
        trip.set("depart", str(depart))
        trip.set("from", from_edge)
        trip.set("to", to_edge)

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

    def get_trips(self):
        # Get all trips as a list of dictionaries
        trips = []
        for trip in self.root.findall("trip"):
            trips.append({
                "id": trip.get("id"),
                "depart": trip.get("depart"),
                "from": trip.get("from"),
                "to": trip.get("to"),
            })
        return trips


class RoutesXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty routes element
            self.root = etree.Element("routes")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/routes_file.xsd")
            self.tree = etree.ElementTree(self.root)

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_routes(self):
        routes = []
        for vehicle in self.root.findall("vehicle"):
            for route in vehicle.findall("route"):
                routes.append({
                    "id": vehicle.get("id"),
                    "depart": vehicle.get("depart"),
                    "edges": route.get("edges")
                })
        return routes


class SUMOTrips(object):
    def __init__(self, sumo_network: SUMONetwork):
        self.sumo_network = sumo_network

    def read_routes(self):
        routes_xml = RoutesXML(self.routes_path)
        return routes_xml.get_routes()

    @property
    def routes_path(self):
        return self.sumo_network.file_prefix + 'routes.rou.xml'

    def generate_lsoa_od_routes(self, area_od_matrix: Optional[AreaOriginDestinationMatrix] = None) -> TripsXML:
        """
            1) find jupyter where I have made LSOA trips
            2) algorithm:
                2.1) take LSOA pair from OD matrix
                2.2) find random point proportional to street length in each of from/to LSOAs
                    - make list of all street lengths [street_0_length, street_1_length, ...]
                    - get streets in the LSOA
                    - flip a coin (select random street)
                    - return street node id (50% cases start node, 50% end node)
                2.3) make trip from selected points
        """
        if area_od_matrix is None:
            area_od_matrix = AreaOriginDestinationMatrix()

        streets_gdf = self.sumo_network.make_streets_gdf()

        lsoa_codes = set(streets_gdf[gis_utils.LSOA_CODE_COLUMN])
        lsoa_code_to_street_index = {lsoa_code: streets_gdf[gis_utils.LSOA_CODE_COLUMN] == lsoa_code for lsoa_code in lsoa_codes}

        od_gm_df = area_od_matrix.load()

        depart = 0
        trips = []
        for i in range(len(od_gm_df)):
            od_record = od_gm_df.iloc[i]
            lsoa_from, lsoa_to = od_record.geo_code1, od_record.geo_code2
            street_indices_from = lsoa_code_to_street_index.get(lsoa_from, None)
            street_indices_to = lsoa_code_to_street_index.get(lsoa_to, None)
            if street_indices_from is None or street_indices_to is None:
                continue
            lsoa_from_edges = streets_gdf[street_indices_from]['edge_id']
            lsoa_to_edges = streets_gdf[street_indices_to]['edge_id']

            routes_count = od_record['all']
            departs = depart + np.random.normal(10, 5, routes_count)
            for j in range(routes_count):
                # TODO: sample streets by its length
                edge_from = lsoa_from_edges.iloc[np.random.randint(0, len(lsoa_from_edges))]
                edge_to = lsoa_to_edges.iloc[np.random.randint(0, len(lsoa_to_edges))]

                trips.append((f"{i}#{j}", max(0, departs[j]), edge_from, edge_to))

        trips_xml = TripsXML()
        # SUMO requirement: Route file should be sorted by departure time
        trips = sorted(trips, key=lambda x: x[1])
        for trip in trips:
            trips_xml.add_trip(*trip)
        trips_path = self.sumo_network.file_prefix + 'trips.rou.xml'
        trips_xml.save(trips_path)
        return trips_xml

    def run_duorouter(self, force=False) -> str:
        # old lsoa example: "duarouter -n gmdata.nosync/lsoa_net.net.xml -t gmdata.nosync/lsoa_trips_od.rou.xml -o gmdata.nosync/lsoa_od_raw_routes.rou.xml --ignore-errors 2&> gmdata.nosync/duarouter.log"

        if os.path.exists(self.routes_path) and not force:
            return self.routes_path

        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
        trips_path = self.sumo_network.file_prefix + 'trips.rou.xml'
        if not os.path.exists(trips_path):
            self.generate_lsoa_od_routes()

        net_path = self.sumo_network.make_net_path()
        cmd = [
            "duarouter",
            "-n",
            net_path,
            "--route-files",
            trips_path,
            "-o",
            self.routes_path,
            "--ignore-errors"
        ]
        subprocess.run(cmd)
        return self.routes_path
