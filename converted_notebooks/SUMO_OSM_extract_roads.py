#!/usr/bin/env python
# coding: utf-8

from bim.gis.traffic.od_matrix import AreaOriginDestinationMatrix
from bim.gis import utils as gis_utils
import geopandas as gpd
import pghstore
import pandas as pd

def extract_tags(other_tags: str):
    if other_tags and not pd.isna(other_tags):
        other_tags = ','.join(list(filter(lambda x: x and x.count('"') == 4 and '=>' in x, other_tags.split(','))))
        try:
            tags = pghstore.loads(other_tags)
            return tags
        except ValueError as e:
            print(f'{other_tags=}', e)
    return None


def extract_max_speed(other_tags: str, default_max_speed: int = 50):
    tags = extract_tags(other_tags)
    if tags is not None:
        maxspeed_str = tags.get('maxspeed', None)
        if maxspeed_str is not None:
            try:
                maxspeed = int(maxspeed_str.split(' ')[0])
                if maxspeed > 0 and maxspeed < 200:
                    return maxspeed
            except ValueError as e:
                print(f'{maxspeed_str=}', e)
    return default_max_speed


def extract_lanes_number(other_tags: str, default_lanes_number: int = 1):
    if other_tags and not pd.isna(other_tags):
        other_tags = ','.join(list(filter(lambda x: x and x.count('"') == 4 and '=>' in x, other_tags.split(','))))
        try:
            tags = pghstore.loads(other_tags)
            return int(tags.get('lanes', default_lanes_number))
        except ValueError as e:
            print(f'{other_tags=}', e)
    return default_lanes_number


OSM_extracted_roads_gdf = gpd.read_file('gmdata.nosync/GM_roads_1_extracted/GM_roads_1_extracted.shp')
OSM_extracted_roads_gdf['lanes_number'] = OSM_extracted_roads_gdf['other_tags'].apply(extract_lanes_number)
OSM_extracted_roads_gdf['max_speed'] = OSM_extracted_roads_gdf['other_tags'].apply(extract_max_speed)

good_highway_types = [
    'service', 'residential', 'unclassified', 'trunk', 'tertiary', 'primary', 'secondary', 'motorway', 'road', 'services',
    'trunk_link', 'motorway_link', 'primary_link', 'tertiary_link', 'living_street', 'secondary_link', 'busway'
]

# from bim.gis import osm
# streets_gdf = osm.Nominatim().load_streets_gdf()
# total_bounds = streets_gdf.geometry.total_bounds
total_bounds = [353784.67760563, 381371.87809096, 404303.27216495, 421261.44411209]
xmin, ymin, xmax, ymax = total_bounds

filtered_OSM_extracted_roads_gdf = OSM_extracted_roads_gdf[OSM_extracted_roads_gdf.highway.isin(good_highway_types)]
filtered_OSM_extracted_roads_gdf = filtered_OSM_extracted_roads_gdf.cx[xmin:xmax, ymin:ymax]

new_streets_gdf = filtered_OSM_extracted_roads_gdf[['osm_id', 'name', 'geometry', 'lanes_number', 'max_speed']].copy()
new_streets_gdf.columns = ['id', 'name', 'geometry', 'num_lanes', 'max_speed']
new_streets_gdf.reset_index(drop=True, inplace=True)
area_od_matrix = AreaOriginDestinationMatrix()
new_streets_gdf = new_streets_gdf.to_crs(gis_utils.EPSG_WSG84)
new_streets_gdf[gis_utils.LSOA_CODE_COLUMN] = area_od_matrix.streets_geometry_to_lsoa(new_streets_gdf)
new_streets_gdf = new_streets_gdf.to_crs(gis_utils.EPSG_BNG)
new_streets_gdf = new_streets_gdf.set_crs(gis_utils.EPSG_BNG, inplace=True)
new_streets_gdf = new_streets_gdf.to_crs(epsg=gis_utils.EPSG_WSG84)
gis_utils.save_geodf_to_csv(new_streets_gdf, 'gmdata.nosync/greater_manchester_streets_with_lsoa_fixed.csv')
