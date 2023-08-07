import pandas as pd
from typing import Iterable
import itertools
import numpy as np


class AreaOriginDistanceMatrix(object):
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

    def load(self) -> pd.DataFrame:
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

    def find_connected_by_streets(self, streets, gm_lsoa_boundaries_gdf):
        from shapely.geometry import Point

        def get_area_code(lsoas_sindex, gm_lsoa_boundaries_gdf):
            def f(coords: Iterable) -> str:
                lon, lat = coords
                p = Point(lon, lat)

                possible_matches_index = list(lsoas_sindex.nearest(p))
                possible_matches = gm_lsoa_boundaries_gdf.iloc[np.asarray(possible_matches_index)[:, 0]]
                precise_matches = possible_matches[possible_matches.intersects(p)]

                # If there are any precise matches, store the LSOA code of the first one
                if not precise_matches.empty:
                    return precise_matches.iloc[0]['geo_code']
                return ''
            return f

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
        return connected
