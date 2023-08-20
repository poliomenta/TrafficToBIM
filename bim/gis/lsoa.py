import geopandas as gpd
import numpy as np
from ..gis.utils import EPSG_WSG84, get_epsg_number, convert_epsg


class LSOABoundaries(object):
    instance = None  # Singleton pattern

    def __init__(self, shp_path: str = 'gmdata.nosync/infuse_lsoa_lyr_2011.shp'):
        """
        source:
            https://statistics.ukdataservice.ac.uk/dataset/2011-census-geography-boundaries-lower-layer-super-output-areas-and-data-zones
        default path columns:
            'geo_code', 'geo_label', 'geo_labelw', 'label', 'name', 'geometry'
        default file rows count:
            42619
        """
        self.shp_path = shp_path
        self.gdf = None

    @staticmethod
    def get_instance():
        """
        Singleton pattern
        """
        if not LSOABoundaries.instance:
            LSOABoundaries.instance = LSOABoundaries()
        return LSOABoundaries.instance

    def load(self, epsg: int = EPSG_WSG84, force: bool = False) -> gpd.GeoDataFrame:
        if self.gdf is None or force:
            self.gdf = gpd.read_file(self.shp_path)
        if epsg:
            self.gdf = convert_epsg(self.gdf, epsg)
        return self.gdf

    def get_epsg_number(self):
        self.load()
        return get_epsg_number(self.gdf)

    def get_geo_codes(self) -> gpd.GeoSeries:
        gdf = self.load()
        return gdf['geo_code']

    def get_lsoa(self, code):
        gdf = self.load()
        return gdf[gdf['geo_code'] == code]

    def plot_random_n_lsoa(self, n=2000):
        import matplotlib.pylab as plt
        self.load()
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        gm_boundary = CountiesBoundary.get_instance().get_gm_boundary()

        ix = np.arange(len(self.gdf))
        np.random.shuffle(ix)
        ix = ix[:n]
        self.gdf.iloc[ix].plot(ax=ax, alpha=0.5)
        gm_boundary.plot(ax=ax, alpha=0.2)


class CountiesBoundary(object):
    instance = None  # Singleton pattern

    def __init__(self, shp_path: str = 'gmdata.nosync/Boundary-line-ceremonial-counties_region.shp'):
        """
        default path columns:
            'NAME', 'DESCRIPTIO', 'geometry'
        default file rows count:
            91
        """
        self.shp_path = shp_path
        self.gdf = None

    @staticmethod
    def get_instance():
        """
        Singleton pattern
        """
        if not CountiesBoundary.instance:
            CountiesBoundary.instance = CountiesBoundary()
        return CountiesBoundary.instance

    def load(self, epsg: int = EPSG_WSG84, force: bool = False) -> gpd.GeoDataFrame:
        if self.gdf is None or force:
            self.gdf = gpd.read_file(self.shp_path)
        if epsg:
            self.gdf = convert_epsg(self.gdf, epsg)
        return self.gdf

    def get_epsg_number(self):
        self.load()
        return get_epsg_number(self.gdf)

    def get_gm_boundary(self, epsg: int = EPSG_WSG84) -> gpd.GeoDataFrame:
        gdf = self.load()
        manchester_gdf = gdf[gdf['NAME'].apply(lambda x: 'manchester' in x.lower())]
        return convert_epsg(manchester_gdf, epsg)

    def get_gm_boundary_polygon(self, epsg: int = EPSG_WSG84):
        gm_boundary = self.get_gm_boundary(epsg)
        return gm_boundary['geometry'].iloc[0]
