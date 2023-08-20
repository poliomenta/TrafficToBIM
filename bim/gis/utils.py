import os.path
import pyarrow
import geopandas as gpd
import pandas as pd
from shapely import wkt


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
    """
    Save a GeoDataFrame to CSV.

    Parameters:
    - geodf: GeoDataFrame
        The GeoDataFrame to save.
    - filename: str
        The path to save the CSV file.
    """
    # Convert geometry to WKT format
    geodf['geometry'] = geodf['geometry'].apply(lambda x: x.wkt)

    # Save to CSV
    geodf.to_csv(filename, index=False)


def read_csv_to_geodf(filename):
    """
    Read a CSV file into a GeoDataFrame.

    Parameters:
    - filename: str
        The path of the CSV file to read.

    Returns:
    - GeoDataFrame
    """
    # Read CSV file
    df = pd.read_csv(filename)

    # Convert WKT string back to a geometry object
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # Convert DataFrame to GeoDataFrame
    geodf = gpd.GeoDataFrame(df, geometry='geometry')

    return geodf
