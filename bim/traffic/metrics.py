
from collections import defaultdict
import os
import sys
from xml.sax.handler import ContentHandler
from xml.sax import make_parser
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel

from .trips import SUMOTrips

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
#     sys.exit("please declare environment variable 'SUMO_HOME'")


RAW_EDGE_METRIC_TRAVELTIME = 'traveltime'
RAW_EDGE_METRIC_TIMELOSS = 'timeLoss'
RAW_EDGE_METRIC_OVERLAP_TRAVELTIME = 'overlapTraveltime'
RAW_EDGE_METRIC_SAMPLED_SECONDS = 'sampledSeconds'

# metrics for optimisation should be as low as possible, because we sort experiments by metric in ascending order
# see update_top_experiments() in OptimiserDumpsManager.run_optimisation() in optimiser.py
EDGE_TOTAL_TRAVELTIME = 'total_traveltime'
EDGE_TOTAL_TIMELOSS = 'total_timeloss'
EDGE_AVG_TRAVELTIME = 'avg_traveltime'
EDGE_AVG_TIMELOSS = 'avg_timeloss'
EDGE_TOTAL_TRIPS_COUNT = 'total_trips_count'


class ExperimentMetrics(BaseModel):
    total_traveltime: float
    avg_traveltime: float
    total_timeloss: float
    total_trips_count: int

    def get(self, metric_name: str):
        return self.dict()[metric_name]

    @classmethod
    def make(cls, raw_metrics, sumo_trips: SUMOTrips):
        total_traveltime = raw_metrics[RAW_EDGE_METRIC_TRAVELTIME].sum()
        total_timeloss = raw_metrics[RAW_EDGE_METRIC_TIMELOSS].sum()
        routes = sumo_trips.read_routes()
        total_trips_count = len(routes)
        metrics = {
            EDGE_TOTAL_TRAVELTIME: total_traveltime,
            EDGE_AVG_TRAVELTIME: total_traveltime / total_trips_count if total_trips_count > 0 else 0,
            EDGE_TOTAL_TIMELOSS: total_timeloss,
            EDGE_AVG_TIMELOSS: total_timeloss / total_trips_count if total_trips_count > 0 else 0,
            EDGE_TOTAL_TRIPS_COUNT: total_trips_count
        }
        return cls(**metrics)

def parse_sax(xmlfile, handler):
    myparser = make_parser()
    myparser.setContentHandler(handler)
    res = myparser.parse(xmlfile)
    return myparser, res


class MetricsReader(ContentHandler):
    """
    Reads metrics from *_aggregated_info.xml files
    """
    def __init__(self, metric_names):
        super().__init__()
        self.metric_values = defaultdict(dict)
        self.metric_names = metric_names

    def startElement(self, name, attrs):
        if name == 'edge':
            edge_id = attrs['id']
            for metric in self.metric_names:
                if metric in attrs:
                    self.metric_values[metric][edge_id] = float(attrs[metric])


def parse_attribute_stats_txt_row(row, metric_name):
    pair = row.replace('\n', '').split(' ')
    metric_value = float(pair[0])
    trip_id = pair[1]
    return {
        metric_name: metric_value,
        'trip_id': f'edge{trip_id}'
    }


def load_raw_trip_metrics_from_attribute_stats(agg_tripinfo_full_path, metric_name='timeLoss'):
    # on incomplete run one can use the following script (example):
    # python $SUMO_HOME/tools/output/attributeStats.py --xml-output agg_tripinfo.xml --full-output agg_tripinfo_full.txt --attribute timeLoss gm_run0.tripinfo.xml

    with open(agg_tripinfo_full_path, 'r') as file:
        agg_tripinfo_full = file.readlines()

    agg_trip_metrics = [parse_attribute_stats_txt_row(row, metric_name) for row in agg_tripinfo_full]
    agg_trip_metrics_df = pd.DataFrame(agg_trip_metrics).set_index('trip_id')
    agg_trip_metrics_df.index.name = None
    return agg_trip_metrics_df


def load_raw_metrics(aggregated_info_file_path: str, metrics: Optional[List[str]] = None, drop_empty: bool = False) \
        -> pd.DataFrame:
    if metrics is None:
        metrics = [
            RAW_EDGE_METRIC_TRAVELTIME,
            RAW_EDGE_METRIC_TIMELOSS,
            RAW_EDGE_METRIC_OVERLAP_TRAVELTIME,
            RAW_EDGE_METRIC_SAMPLED_SECONDS
        ]
    metrics_reader = MetricsReader(metrics)
    try:
        parse_sax(aggregated_info_file_path, metrics_reader)
    except Exception as e:
        print(e)
    metric_values = metrics_reader.metric_values
    metrics_df = pd.DataFrame(metric_values)
    if drop_empty:
        metrics_df.dropna(inplace=True, how='any')
    return metrics_df
