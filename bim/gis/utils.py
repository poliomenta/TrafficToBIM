import os.path
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely import affinity
from shapely.ops import split as shapely_split
from shapely.geometry import LineString


EPSG_BNG = 27700
EPSG_WSG84 = 4326

LSOA_CODE_COLUMN = 'LSOA_code'


def get_epsg_number(gdf: gpd.GeoDataFrame) -> int:
    return gdf.crs.to_epsg()


def convert_epsg(gdf: gpd.GeoDataFrame, to_epsg) -> gpd.GeoDataFrame:
    epsg = get_epsg_number(gdf)
    if to_epsg is not None and epsg != to_epsg:
        return gdf.to_crs(epsg=to_epsg)
    return gdf


def split_linestring(ls, position=0.5):
    point = ls.interpolate(position, True)
    ls_split = LineString([affinity.translate(point, -0.001, 0), affinity.translate(point, 0.001, 0)])
    result = shapely_split(ls, ls_split)
    return list(result.geoms)


def replace_linestring_coordinate(linestring, coord_index, point):
    coords = list(linestring.coords)
    coords[coord_index] = (point.x, point.y)
    return LineString(coords)


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def cached_pd(file_prefix, verbose=True):
    def decorator(func):
        engine = 'pyarrow'

        def new_func(*args):
            params_tuple = sorted(tuple(args))
            file_path = file_prefix + '_'.join(map(str,params_tuple)) + '.parquet'
            file_path = os.path.abspath(file_path)
            if os.path.exists(file_path):
                if verbose:
                    print(f'use cached file: {file_path}')
                return pd.read_parquet(file_path, engine=engine)
            print(f'cache file path: {file_path}')
            df: pd.DataFrame = func(*args)
            df.to_parquet(file_path, engine=engine)
            return df

        return new_func

    return decorator


def save_geodf_to_csv(geodf, filename):
    geodf['geometry'] = geodf['geometry'].apply(lambda x: x.wkt)
    geodf.to_csv(filename, index=False)


def read_csv_to_geodf(filename):
    df = pd.read_csv(filename)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    geodf = gpd.GeoDataFrame(df, geometry='geometry')
    return geodf
