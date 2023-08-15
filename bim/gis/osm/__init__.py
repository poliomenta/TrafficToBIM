import os.path
import re

from ..traffic.od_matrix import AreaOriginDestinationMatrix
from ...gis import utils as gis_utils
import geopandas as gpd
from shapely.geometry import LineString


class Nominatim(object):
    def parse_geometry(self, geom_str):
        match = re.match(r"LINESTRING\((.+)\)", geom_str.strip())
        if match:
            return list(map(lambda x: x.split(' '), match.group(1).split(',')))
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
        # TODO: wrap streets to List of objects
        # also 'manchester_m60.txt' and 'manchester_streets_north_west.txt' dumps exist
        with open(path, 'r') as f:
            streets = list(filter(None, map(self.parse_street, f.readlines())))
            return streets

    def load_streets_gdf(self, gdf_path='gmdata.nosync/greater_manchester_streets_with_lsoa.csv',
                         original_path='gmdata.nosync/greater_manchester_streets.txt',
                         crs: int = gis_utils.EPSG_WSG84) -> gpd.GeoDataFrame:
        if not os.path.exists(gdf_path):
            gdf = self.make_streets_with_lsoa_gdf(original_path, crs)
            gdf.to_csv(gdf_path, index=False)
            gis_utils.save_geodf_to_csv(gdf, gdf_path)

        gdf = gis_utils.read_csv_to_geodf(gdf_path)
        gdf.set_crs(crs, inplace=True)
        return gdf

    def make_streets_with_lsoa_gdf(self,
                                   original_path='gmdata.nosync/greater_manchester_streets.txt',
                                   crs: int = gis_utils.EPSG_WSG84):
        streets = self.load_streets(original_path)
        streets_gdf = gpd.GeoDataFrame(
            streets,
            columns=['id', 'name', 'coords'],
            geometry=[LineString(street[2]) for street in streets]
        )
        streets_gdf.crs = crs

        area_od_matrix = AreaOriginDestinationMatrix()
        streets_gdf[gis_utils.LSOA_CODE_COLUMN] = area_od_matrix.streets_geometry_to_lsoa(streets_gdf)
        return streets_gdf


def test_parse_street():
    res = Nominatim().parse_street(
        """(554321812, 'Bolton Road', 'LINESTRING(-2.3815276 53.5418358,-2.3817707 53.5419047,-2.3818233 53.5419195,-2.3822453 53.542038,-2.3826206 53.5421584,-2.3826765 53.5421764,-2.382974 53.5422726,-2.3833405 53.5423939,-2.3849496 53.5429316,-2.3850697 53.5429756)')""")
    true_res = (554321812,
                'Bolton Road',
                [['-2.3815276', '53.5418358'],
                 ['-2.3817707', '53.5419047'],
                 ['-2.3818233', '53.5419195'],
                 ['-2.3822453', '53.542038'],
                 ['-2.3826206', '53.5421584'],
                 ['-2.3826765', '53.5421764'],
                 ['-2.382974', '53.5422726'],
                 ['-2.3833405', '53.5423939'],
                 ['-2.3849496', '53.5429316'],
                 ['-2.3850697', '53.5429756']])
    assert res == true_res
