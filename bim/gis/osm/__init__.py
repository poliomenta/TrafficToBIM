import os.path
import re

from ..traffic.od_matrix import AreaOriginDestinationMatrix
from ...gis import utils as gis_utils
import geopandas as gpd
from shapely.geometry import LineString, Polygon


OLD_STREETS_GDF_PATH = 'gmdata.nosync/greater_manchester_streets_with_lsoa.csv'
FIXED_STREETS_GDF_PATH = 'gmdata.nosync/greater_manchester_streets_with_lsoa_fixed.csv'


class Nominatim(object):
    def parse_geometry(self, geom_str):
        def clean(s):
            return s.replace('(', '').replace(')', '')
        match = re.match(r"LINESTRING\((.+)\)", geom_str.strip())
        if match:
            return list(map(lambda x: x.split(' '), match.group(1).split(',')))

        match = re.match(r"POLYGON\(\((.+)\)\)", geom_str.strip())
        if match:
            return list(map(lambda x: clean(x).split(' '), match.group(1).split(',')))

        match = re.match(r"POINT\((.+)\)", geom_str.strip())
        if match:
            return [match.group(1).split(' ')]
        return None

    def parse_street(self, street_txt: str, verbose: bool = False):
        street_txt = street_txt.replace('None', "'None'")
        match = re.match(r"\((\d+), ('|\")(.+)('|\"), '(.+)'\)", street_txt.strip())
        if match:
            osm_id = int(match.group(1))
            name = match.group(3)
            linestring_wkt = self.parse_geometry(match.group(5))
            if linestring_wkt is None or len(linestring_wkt) == 1:
                if verbose:
                    print(street_txt)
                return None
            #         linestring = wkt.loads(linestring_wkt)
            return osm_id, name, linestring_wkt
        return None

    def load_streets(self, path='gmdata.nosync/greater_manchester_streets.txt'):
        # better to wrap streets to List of objects
        with open(path, 'r') as f:
            streets = list(filter(None, map(self.parse_street, f.readlines())))
            return streets

    def load_streets_gdf(self, gdf_path=FIXED_STREETS_GDF_PATH,
                         original_path='gmdata.nosync/greater_manchester_streets.txt',
                         crs: int = gis_utils.EPSG_WSG84,
                         polygon: bool = False) -> gpd.GeoDataFrame:
        """
        default gdf columns:
        - id (osm_id)
        - name
        - coords (WKT)
        - geometry (shapely.geometry.Linestring)
        - LSOA_code (string)
        for FIXED_STREETS_GDF_PATH also:
        - max_speed
        - num_lanes
        """
        if not os.path.exists(gdf_path):
            gdf = self.make_streets_with_lsoa_gdf(original_path, crs, polygon)
            # gdf.to_csv(gdf_path, index=False)
            gis_utils.save_geodf_to_csv(gdf, gdf_path)

        gdf = gis_utils.read_csv_to_geodf(gdf_path)
        gdf.set_crs(crs, inplace=True)
        return gdf

    def make_streets_with_lsoa_gdf(self,
                                   original_path='gmdata.nosync/greater_manchester_streets.txt',
                                   crs: int = gis_utils.EPSG_WSG84,
                                   polygon: bool = False):
        streets = self.load_streets(original_path)
        if polygon:
            streets_gdf = gpd.GeoDataFrame(
                streets,
                columns=['id', 'name', 'coords'],
                geometry=[Polygon(street[2]) for street in streets]
            )
        else:
            streets_gdf = gpd.GeoDataFrame(
                streets,
                columns=['id', 'name', 'coords'],
                geometry=[LineString(street[2]) for street in streets]
            )
        streets_gdf.crs = crs

        if not polygon:  # no need
            area_od_matrix = AreaOriginDestinationMatrix()
            streets_gdf[gis_utils.LSOA_CODE_COLUMN] = area_od_matrix.streets_geometry_to_lsoa(streets_gdf)
        return streets_gdf

