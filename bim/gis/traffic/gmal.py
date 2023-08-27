import pandas as pd


class GMAL(object):
    def __init__(self, csv_path: str = 'gmdata.nosync/GMAL+Population+Bad_areas_2021.csv'):
        """
        view map online:
            https://mappinggm.org.uk/gmodin/?lyrs=v_gm_bridges_structures,tfgm_bus_stops,v_tfgm_bus_routes,v_sustrans_cycle_network,rail_network,tfgm_gmal#os_maps_light/11/53.5118/-2.2854
        source:
            https://www.data.gov.uk/dataset/d9dfbf0a-3cd7-4b12-a39f-0ec717423ee4/gm-accessibility-levels-gmal
        default path columns:
            'area_code', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4',
            'Unnamed: 5', 'area_name', 'census_geography', 'all_residents_2011',
            'area_ha', 'density_2011', 'centroid_x', 'centroid_y',
            'is_bad_area_2011', 'Unnamed: 14', 'Population_2021', 'density_2021',
            'is_bad_area_2021', 'Unnamed: 18', 'population_2011_2021, %',
            'gmal_pos_formula', 'gmal_pos_value', 'gmal_score',
            'is_bad_area_2011.1', 'GMAL_threshold', 'Unnamed: 25',
            'bad_area_ratio, %', 'bad_area_ratio, %.1', 'Population_threshold_2011',
            'Population_threshold_2021', 'GMAL_min', 'GMAL_max', 'area_name_2'
        """
        self.csv_path = csv_path
        self.df = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        return self.df
