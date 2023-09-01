import pandas as pd

from ..gis.traffic.od_matrix import AreaOriginDestinationMatrix
from ..sumo.network import SUMONetwork
from ..gis import utils as gis_utils
from typing import Optional
import numpy as np
import subprocess
import os
from lxml import etree


def add_suffix_to_filepath(filepath, suffix):
    directory, filename_with_exts = os.path.split(filepath)
    first_period_index = filename_with_exts.find(".")
    if first_period_index == -1:
        new_filename = f"{filename_with_exts}_{suffix}"
    else:
        filename = filename_with_exts[:first_period_index]
        extensions = filename_with_exts[first_period_index:]
        new_filename = f"{filename}_{suffix}{extensions}"
    new_filepath = os.path.join(directory, new_filename)
    return new_filepath


class TripsXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            self.root = etree.Element("routes")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/routes_file.xsd")
            self.tree = etree.ElementTree(self.root)

    def add_trip(self, id, depart, from_edge, to_edge):
        trip = etree.SubElement(self.root, "trip")
        trip.set("id", id)
        trip.set("depart", str(depart))
        trip.set("from", from_edge)
        trip.set("to", to_edge)

    def save(self, path):
        self.tree.write(path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def tostring(self):
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()

    def load(self, xml_file):
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_trips(self):
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
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            self.root = etree.Element("routes")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/routes_file.xsd")
            self.tree = etree.ElementTree(self.root)

    def tostring(self):
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()

    def load(self, xml_file):
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def save(self, path):
        self.tree.write(path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def add_route(self, id: str, depart: float, edges: str):
        vehicle = etree.SubElement(self.root, "vehicle")
        vehicle.set("id", id)
        vehicle.set("depart", str(depart))
        route = etree.SubElement(vehicle, "route")
        route.set("edges", edges)

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

    def update_routes(self, routes_df: pd.DataFrame):
        """
        :param routes_df: columns: 'id', 'depart', 'edges'
        """
        os.makedirs(os.path.dirname(self.routes_path), exist_ok=True)

        routes_xml = RoutesXML()
        for t in routes_df.itertuples():
            routes_xml.add_route(t.id, t.depart, t.edges)
        return routes_xml.save(self.routes_path)

    def update_trips(self, routes_df):
        """
        :param routes_df: columns: 'id', 'depart', 'edges'
        """
        trips_xml = TripsXML()
        for t in routes_df.itertuples():
            edges = t.edges.split(' ')
            if edges[0] == edges[-1]:
                continue
            trips_xml.add_trip(t.id, t.depart, edges[0], edges[-1])
        trips_xml.save(self.trips_path)

    def reduce_car_trips(self, original_trips_path: str, trips_path: str, reduce_percent: float):
        original_trips_xml = TripsXML(original_trips_path)
        trips = original_trips_xml.get_trips()
        np.random.shuffle(trips)
        perc = max(0., 1 - reduce_percent)
        trips = trips[:int(perc * len(trips))]
        # SUMO requirement: Route file should be sorted by departure time
        new_trips_xml = TripsXML()
        trips = sorted(trips, key=lambda x: x['depart'])
        for t in trips:
            new_trips_xml.add_trip(t['id'], t['depart'], t['from'], t['to'])
        new_trips_xml.save(trips_path)

    @property
    def routes_path(self):
        return self.sumo_network.file_prefix + 'routes.rou.xml'

    @property
    def trips_path(self):
        return self.sumo_network.file_prefix + 'trips.rou.xml'

    def generate_lsoa_od_routes(self, area_od_matrix: Optional[AreaOriginDestinationMatrix] = None) -> TripsXML:
        """
            algorithm:
                1) take LSOA pair from OD matrix
                2) find random point proportional to street length in each of from/to LSOAs
                    - make list of all street lengths [street_0_length, street_1_length, ...]
                    - get streets in the LSOA
                    - flip a coin (select random street)
                    - return street node id (50% cases start node, 50% end node)
                3) make trip from selected points
        """
        print('generate routes with LSOA OD-matrix')
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
                # better to sample streets by its length
                edge_from = lsoa_from_edges.iloc[np.random.randint(0, len(lsoa_from_edges))]
                edge_to = lsoa_to_edges.iloc[np.random.randint(0, len(lsoa_to_edges))]

                trips.append((f"{i}#{j}", max(0, departs[j]), edge_from, edge_to))

        trips_xml = TripsXML()
        # SUMO requirement: Route file should be sorted by departure time
        trips = sorted(trips, key=lambda x: x[1])
        for trip in trips:
            trips_xml.add_trip(*trip)
        trips_path = self.trips_path
        trips_xml.save(trips_path)
        return trips_xml

    def run_duarouter(self, reduce_cars_by_percent: float = 0., force=False) -> str:
        # old lsoa example: "duarouter -n gmdata.nosync/lsoa_net.net.xml -t gmdata.nosync/lsoa_trips_od.rou.xml -o gmdata.nosync/lsoa_od_raw_routes.rou.xml --ignore-errors 2&> gmdata.nosync/duarouter.log"

        if os.path.exists(self.routes_path) and not force:
            return self.routes_path

        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'

        trips_path = self.trips_path
        if not os.path.exists(self.trips_path):
            self.generate_lsoa_od_routes()  # better to support reduce_cars_by_percent here
        elif reduce_cars_by_percent > 0:
            reduced_trips_path = add_suffix_to_filepath(self.trips_path, '_reduced')
            self.reduce_car_trips(self.trips_path, reduced_trips_path, reduce_cars_by_percent)
            trips_path = reduced_trips_path

        print('run duarouter')
        net_path = self.sumo_network.make_net_path()
        cmd = [
            "duarouter",
            "-n",
            net_path,
            "--route-files",
            trips_path,
            "--repair",
            "-o",
            self.routes_path,
            "--ignore-errors"
        ]
        subprocess.run(cmd)
        return self.routes_path
