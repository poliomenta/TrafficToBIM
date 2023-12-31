
from collections import defaultdict
import os
import sys
from xml.sax.handler import ContentHandler
from xml.sax import make_parser
from typing import Optional, List, Tuple
import pandas as pd
from pydantic import BaseModel

from .trips import SUMOTrips

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
    sys.exit("please declare environment variable 'SUMO_HOME'")


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

EDGE_TOTAL_BUS_TRAVELTIME = 'total_bus_traveltime'
EDGE_TOTAL_BUS_TIMELOSS = 'total_bus_timeloss'
EDGE_AVG_BUS_TRAVELTIME = 'avg_bus_traveltime'
EDGE_AVG_BUS_TIMELOSS = 'avg_bus_timeloss'
EDGE_TOTAL_BUS_TRIPS_COUNT = 'total_bus_trips_count'

FAILED_METRIC_VALUE = float('inf')


class ExperimentMetrics(BaseModel):
    total_traveltime: float
    avg_traveltime: float
    total_timeloss: float
    total_trips_count: int
    total_bus_traveltime: Optional[float] = None
    total_bus_timeloss: Optional[float] = None
    avg_bus_traveltime: Optional[float] = None
    total_bus_trips_count: Optional[int] = None

    def get(self, metric_name: str):
        return self.dict()[metric_name]

    @staticmethod
    def make_failed():
        return ExperimentMetrics(total_traveltime=FAILED_METRIC_VALUE, avg_traveltime=FAILED_METRIC_VALUE,
                                 total_timeloss=-FAILED_METRIC_VALUE, total_trips_count=0)

    @classmethod
    def make(cls, raw_metrics, raw_bus_metrics, sumo_trips: SUMOTrips):
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

        total_bus_trips_count = len(raw_bus_metrics)
        bus_metrics = {}
        if total_bus_trips_count > 0 and RAW_EDGE_METRIC_TRAVELTIME in raw_bus_metrics.columns:
            total_bus_traveltime = raw_bus_metrics[RAW_EDGE_METRIC_TRAVELTIME].sum()
            total_bus_timeloss = raw_bus_metrics[RAW_EDGE_METRIC_TIMELOSS].sum()
            bus_metrics = {
                EDGE_TOTAL_BUS_TRAVELTIME: total_bus_traveltime,
                EDGE_AVG_BUS_TRAVELTIME: total_bus_traveltime / total_bus_trips_count if total_bus_trips_count > 0 else 0,
                EDGE_TOTAL_BUS_TIMELOSS: total_bus_timeloss,
                EDGE_AVG_BUS_TIMELOSS: total_bus_timeloss / total_bus_trips_count if total_bus_trips_count > 0 else 0,
                EDGE_TOTAL_BUS_TRIPS_COUNT: total_bus_trips_count
            }
        return cls(**metrics, **bus_metrics)


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
            if 'vType' in attrs:
                self.metric_values['vType'][edge_id] = attrs['vType']
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


def load_raw_metrics(aggregated_info_file_path: str, aggregated_bus_info_file_path: str,
                     metrics: Optional[List[str]] = None, drop_empty: bool = False) -> Tuple[pd.DataFrame,pd.DataFrame]:
    result = []
    for file in [aggregated_info_file_path, aggregated_bus_info_file_path]:
        if metrics is None:
            metrics = [
                RAW_EDGE_METRIC_TRAVELTIME,
                RAW_EDGE_METRIC_TIMELOSS,
                RAW_EDGE_METRIC_OVERLAP_TRAVELTIME,
                RAW_EDGE_METRIC_SAMPLED_SECONDS,
            ]
        metrics_reader = MetricsReader(metrics)
        try:
            parse_sax(file, metrics_reader)
        except Exception as e:
            print(e)
        metric_values = metrics_reader.metric_values
        metrics_df = pd.DataFrame(metric_values)
        if drop_empty:
            metrics_df.dropna(inplace=True, how='any')
        result.append(metrics_df)
    return result[0], result[1]
