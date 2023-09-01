import os.path

import pandas as pd
import geopandas as gpd
from typing import Iterable, Optional
import itertools
import numpy as np
import json
import matplotlib.pylab as plt
from shapely.geometry import Point

from ..lsoa import CountiesBoundary, LSOABoundaries

LSOA_virtual_streets_connections_path = 'gmdata.nosync/lsoa_connections.json'


def get_area_code(lsoas_sindex, gm_lsoa_boundaries_gdf):
    def f(coords: Iterable) -> str:
        lon, lat = coords
        p = Point(lon, lat)

        possible_matches_index = list(lsoas_sindex.nearest(p))
        # mb here [1, :] better?
        possible_matches = gm_lsoa_boundaries_gdf.iloc[np.asarray(possible_matches_index)[:, 0]]
        precise_matches = possible_matches[possible_matches.intersects(p)]

        # If there are any precise matches, store the LSOA code of the first one
        if not precise_matches.empty:
            return precise_matches.iloc[0]['geo_code']
        return ''

    return f


class AreaOriginDestinationMatrix(object):
    def __init__(self, csv_path: str = 'gmdata.nosync/commute-lsoa-greater-manchester-od_attributes.csv'):
        """
        view map online (for cycling only):
            https://www.pct.bike/m/?r=greater-manchester
        source:
            LSOA-level flows data -> All flows (attribute data only)
            https://www.pct.bike/m/?r=greater-manchester
        default path columns:
            'id',  'geo_code1',  'geo_code2',  'geo_name1',  'geo_name2',  'lad11cd1',  'lad11cd2',
            'lad_name1',  'lad_name2',  'all',  'bicycle',  'foot',  'car_driver',  'car_passenger',  'motorbike',
            'train_tube',  'bus',  'taxi_other',  'govtarget_slc',  'govtarget_sic',  'govtarget_slw',  'govtarget_siw',
            'govtarget_sld',  'govtarget_sid',  'govtarget_slp',  'govtarget_sip',  'govtarget_slm',  'govtarget_sim',
            'govtarget_slpt',  'govtarget_sipt',  'govnearmkt_slc',  'govnearmkt_sic',  'govnearmkt_slw',
            'govnearmkt_siw',  'govnearmkt_sld',  'govnearmkt_sid',  'govnearmkt_slp',  'govnearmkt_sip',
            'govnearmkt_slm',  'govnearmkt_sim',  'govnearmkt_slpt',  'govnearmkt_sipt',  'gendereq_slc',
            'gendereq_sic',  'gendereq_slw',  'gendereq_siw',  'gendereq_sld',  'gendereq_sid',  'gendereq_slp',
            'gendereq_sip',  'gendereq_slm',  'gendereq_sim',  'gendereq_slpt',  'gendereq_sipt',  'dutch_slc',
            'dutch_sic',  'dutch_slw',  'dutch_siw',  'dutch_sld',  'dutch_sid',  'dutch_slp',  'dutch_sip',
            'dutch_slm',  'dutch_sim',  'dutch_slpt',  'dutch_sipt',  'ebike_slc',  'ebike_sic',  'ebike_slw',
            'ebike_siw',  'ebike_sld',  'ebike_sid',  'ebike_slp',  'ebike_sip',  'ebike_slm',  'ebike_sim',
            'ebike_slpt',  'ebike_sipt',  'base_slcyclehours',  'govtarget_sicyclehours',  'govnearmkt_sicyclehours',
            'gendereq_sicyclehours',  'dutch_sicyclehours',  'ebike_sicyclehours',  'base_sldeath',  'base_slyll',
            'base_slvalueyll',  'base_slsickdays',  'base_slvaluesick',  'base_slvaluecomb',  'govtarget_sideath',
            'govtarget_siyll',  'govtarget_sivalueyll',  'govtarget_sisickdays',  'govtarget_sivaluesick',
            'govtarget_sivaluecomb',  'govnearmkt_sideath',  'govnearmkt_siyll',  'govnearmkt_sivalueyll',
            'govnearmkt_sisickdays',  'govnearmkt_sivaluesick',  'govnearmkt_sivaluecomb',  'gendereq_sideath',
            'gendereq_siyll',  'gendereq_sivalueyll',  'gendereq_sisickdays',  'gendereq_sivaluesick',
            'gendereq_sivaluecomb',  'dutch_sideath',  'dutch_siyll',  'dutch_sivalueyll',  'dutch_sisickdays',
            'dutch_sivaluesick',  'dutch_sivaluecomb',  'ebike_sideath',  'ebike_siyll',  'ebike_sivalueyll',
            'ebike_sisickdays',  'ebike_sivaluesick',  'ebike_sivaluecomb',  'base_slcarkm',  'base_slco2',
            'govtarget_sicarkm',  'govtarget_sico2',  'govnearmkt_sicarkm',  'govnearmkt_sico2',  'gendereq_sicarkm',
            'gendereq_sico2',  'dutch_sicarkm',  'dutch_sico2',  'ebike_sicarkm',  'ebike_sico2',  'e_dist_km',
            'rf_dist_km',  'rq_dist_km',  'dist_rf_e',  'dist_rq_rf',  'rf_avslope_perc',  'rq_avslope_perc',
            'rf_time_min',  'rq_time_min'
        """
        self.csv_path = csv_path
        self.df = None

    def load(self, force=False) -> pd.DataFrame:
        if self.df is None or force:
            self.df = pd.read_csv(self.csv_path)
        return self.df

    def get_grouped_geocode_sum(self, attribute: str = 'all') -> pd.DataFrame:
        """
        :param attribute: which attribute to sum, by default 'all' - all kind of transport commuters
        :return:
        """
        if self.df is None:
            self.load()
        return self.df.groupby('geo_code1').agg({attribute: 'sum'}).reset_index()

    def streets_geometry_to_lsoa(self, streets_gdf):
        gm_lsoa_boundaries_gdf = self.get_gm_lsoa_boundaries()
        get_area_code_f = get_area_code(gm_lsoa_boundaries_gdf.sindex, gm_lsoa_boundaries_gdf)
        return streets_gdf.geometry.apply(lambda g: get_area_code_f(g.coords[0]))

    def find_connected_by_streets(self, streets, gm_lsoa_boundaries_gdf, save_path=LSOA_virtual_streets_connections_path):
        connected = set()
        get_area_code_f = get_area_code(gm_lsoa_boundaries_gdf.sindex, gm_lsoa_boundaries_gdf)
        for street_id, street_name, street_coords in streets:
            areas = list(map(get_area_code_f, street_coords))
            # for all pairs of areas which this street connects
            for i, j in itertools.product(areas, areas):
                if i == j:
                    continue
                pair = tuple(sorted([i, j]))
                connected.add(pair)

        if save_path:
            with open(save_path, 'w+') as f:
                json.dump(list(connected), f)
                print(f'Saved to {save_path}')
        return connected

    def get_gm_lsoa_boundaries(self,
                               counties_boundary: Optional[CountiesBoundary] = None,
                               lsoa_boundaries: Optional[LSOABoundaries] = None
                               ) -> gpd.GeoDataFrame:
        if not lsoa_boundaries:
            lsoa_boundaries = LSOABoundaries.get_instance()

        gdf = lsoa_boundaries.load()
        if not counties_boundary:
            counties_boundary = CountiesBoundary.get_instance()

        gm_geocodes = self.get_geocodes()

        gm_boundary_poly = counties_boundary.get_gm_boundary_polygon()
        gm_lsoa_index = gdf.apply(lambda x: x['geo_code'] in gm_geocodes and x['geometry'].within(gm_boundary_poly),
                                  axis=1)
        gm_lsoa_boundaries_gdf = gdf[gm_lsoa_index]
        return gm_lsoa_boundaries_gdf

    # need to test function
    def plot_all_connected_lsoa(self,
                                lsoa_boundaries: Optional[LSOABoundaries] = None,
                                counties_boundary: Optional[CountiesBoundary] = None,
                                connections_path=LSOA_virtual_streets_connections_path,
                                save: bool = True,
                                first_n_connections: Optional[int] = None):


        if os.path.exists(connections_path):
            with open(connections_path, 'r') as file:
                connected = json.load(file)
        else:

            from ...gis import osm
            streets = osm.Nominatim().load_streets()
            gm_lsoa_boundaries_gdf = self.get_gm_lsoa_boundaries()
            connected = self.find_connected_by_streets(streets, gm_lsoa_boundaries_gdf, connections_path)

        if not lsoa_boundaries:
            lsoa_boundaries = LSOABoundaries.get_instance()
        if not counties_boundary:
            counties_boundary = CountiesBoundary.get_instance()

        gm_boundary = counties_boundary.get_gm_boundary()

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        # lsoa_gm_boundary.plot(aspect=1, ax=ax)
        gm_boundary.plot(ax=ax, alpha=0.2)
        connected_list = list(connected)
        if first_n_connections:
            connected_list = connected_list[:first_n_connections]
        for code1, code2 in connected_list:
            if not code1 or not code2:
                continue
            lsoa1 = lsoa_boundaries.get_lsoa(code1)
            lsoa2 = lsoa_boundaries.get_lsoa(code2)
            lsoa1.plot(ax=ax)
            lsoa2.plot(ax=ax)
            lon1, lat1 = list(lsoa1.iloc[0].geometry.centroid.coords)[0]
            lon2, lat2 = list(lsoa2.iloc[0].geometry.centroid.coords)[0]
            plt.plot([lon1, lon2], [lat1, lat2], 'o-')
        if save:
            plt.savefig('connected_lsoa_all.png')
        else:
            plt.show()

    @staticmethod
    def get_vehicle_columns():
        return ['all', 'bicycle', 'foot', 'car_driver', 'car_driver', 'motorbike', 'train_tube', 'bus']

    def plot_od_from(self, from_code: str,
                     lsoa_boundaries: Optional[LSOABoundaries] = None,
                     counties_boundary: Optional[CountiesBoundary] = None,
                     ):
        from_code_routes = self.get_routes_from(from_code)
        from_code_destinations = set(from_code_routes['geo_code2'])

        if not lsoa_boundaries:
            lsoa_boundaries = LSOABoundaries.get_instance()
        if not counties_boundary:
            counties_boundary = CountiesBoundary.get_instance()

        gm_boundary = counties_boundary.get_gm_boundary()
        lsoa_boundaries_gdf = lsoa_boundaries.load()

        lsoa_gm_boundary = lsoa_boundaries_gdf[
            lsoa_boundaries_gdf['geo_code'].apply(lambda x: x in from_code_destinations)]
        gm_boundary_poly = gm_boundary['geometry'].iloc[0]
        lsoa_gm_boundary = lsoa_gm_boundary[
            lsoa_gm_boundary['geometry'].apply(lambda x: x.within(gm_boundary_poly))
        ]
        from_code_routes.loc[:, 'geo_code'] = from_code_routes['geo_code2']
        vehicle_columns = self.get_vehicle_columns()
        lsoa_gm_boundary = lsoa_gm_boundary.merge(from_code_routes[['geo_code'] + vehicle_columns], on='geo_code')

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        lsoa_gm_boundary.plot(aspect=1, ax=ax, column='all', cmap='coolwarm', legend=False)
        lsoa_boundaries_gdf[lsoa_boundaries_gdf['geo_code'] == from_code].plot(aspect=1, ax=ax, color='r')
        gm_boundary.plot(ax=ax, alpha=0.2)

    def plot_top_manchester_routes(self, topn=20):
        top_manch_routes_codes = self.get_top_manchester_routes()
        for i in range(topn):
            self.plot_od_from(top_manch_routes_codes[i])

    def get_geocodes(self):
        df = self.load()
        return set(df['geo_code1'])

    def get_routes_from(self, from_code):
        df = self.load()
        return df[df['geo_code1'] == from_code].copy()

    def get_top_manchester_routes(self):
        df = self.load()
        top_manch_routes = df[(df['lad_name1'] == 'Manchester') &
                                      (df['geo_code2'].apply(lambda x: x.startswith('E')))] \
            .sort_values('all', ascending=False)
        return top_manch_routes['geo_code1'].unique()