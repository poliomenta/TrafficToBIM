"""
folders structure:
gmdata.nosync/
    location_name/ # location name like "StockportCenter" to combine all experiments with that location
        location_name_nodes.nod.xml
        location_name_edges.edg.xml
        location_name_net.net.xml

    location_name_experiments/
        run_0/
            # before sumo run
            StockportCenter_run_0_config.json
            StockportCenter_run_0_net.net.xml
            StockportCenter_run_0_edges.edg.xml
            StockportCenter_run_0_trips.rou.xml
            StockportCenter_run_0_routes.rou.xml
            StockportCenter_run_0_sumo_config.sumocfg
            StockportCenter_run_0_additional_config.xml
            # after sumo run
            StockportCenter_run_0_trip_info.tripinfo.xml
            StockportCenter_run_0_aggregated_info.xml
            ?StockportCenter_run_0_netstate_dump.xml

            !StockportCenter_run_0_model.ifx # includes everything what we need
            !StockportCenter_run_0_metrics.csv # aggregated metrics for the whole network

            images/
                sumo_traffic_nets/
                    sumo_timeloss_map.png
                    sumo_traveltime_map.png
                    sumo_co2_map.png

        run_1
            # before sumo run
            StockportCenter_run_1_config.json
            StockportCenter_run_1_net.net.xml
            StockportCenter_run_1_edges.edg.xml
            StockportCenter_run_1_trips.rou.xml
            StockportCenter_run_1_routes.rou.xml
            StockportCenter_run_1_sumo_config.sumocfg
            StockportCenter_run_1_additional_config.xml
"""
import os
from typing import List, Optional, Set
from pydantic import BaseModel, Extra

from ..traffic.metrics import load_raw_metrics, ExperimentMetrics, EDGE_AVG_TRAVELTIME, \
    load_raw_trip_metrics_from_attribute_stats
from ..traffic.trips import SUMOTrips, RoutesXML
from ..sumo.client import SUMOClient
from ..sumo.network import SUMONetwork
from lxml import etree
import logging
import pandas as pd

logging.basicConfig(filename='network_optimiser', level=logging.INFO)


class SUMOConfigXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            self.root = etree.Element("configuration")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
            self.tree = etree.ElementTree(self.root)

    def configurate_paths(self,
                          route_path: str,
                          net_path: str,
                          tripinfo_output_path: str,
                          netstate_dump_path: str,
                          additional_config_path: str,
                          bus_stops_path: str):
        input = etree.SubElement(self.root, "input")

        route_files = etree.SubElement(input, "route-files")
        route_files.set("value", route_path)

        net_file = etree.SubElement(input, "net-file")
        net_file.set("value", net_path)

        output = etree.SubElement(self.root, "output")

        tripinfo_output = etree.SubElement(output, "tripinfo-output")
        tripinfo_output.set("value", tripinfo_output_path)

        # netstate_dump = etree.SubElement(output, "netstate-dump")
        # netstate_dump.set("value", netstate_dump_path)

        additional_files = etree.SubElement(self.root, "additional-files")
        additional_files.set("value", f'{additional_config_path},{bus_stops_path}')

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True)

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file


class RegionExperiment(BaseModel):
    location_name: str
    run_name: str
    dir_path: str
    base_network_path: str  # path to SUMONetwork json file
    config: Optional[dict]
    metrics: Optional[ExperimentMetrics] = None

    def make_config_hash(self):
        return str(hash((self.base_network_path, str(self.config))))

    @property
    def run_path(self):
        return os.path.join(self.dir_path, self.run_name)

    @property
    def full_run_name(self):
        return self.location_name + '_' + self.run_name

    def generate_sumoconfigs(self, sumo_network: SUMONetwork) -> str:
        additional_config_path = sumo_network.file_prefix + 'additional_config.xml'
        aggregated_info_path = sumo_network.name + '_aggregated_info.xml'
        aggregated_info_xml = f"""
<additional>
    <edgeData id="aggregate" file="{aggregated_info_path}"/>
</additional>
        """
        with open(additional_config_path, 'w+') as file:
            file.write(aggregated_info_xml)

        sumo_config_xml = SUMOConfigXML()
        sumo_config_xml.configurate_paths(
            sumo_network.name + '_routes.rou.xml',
            sumo_network.name + '_net.net.xml',
            sumo_network.name + '.tripinfo.xml',
            sumo_network.name + '_netstate_dump.xml',
            sumo_network.name + '_additional_config.xml',
            sumo_network.name + '_bus_stops.add.xml',
        )
        sumoconfig_path = sumo_network.file_prefix + 'sumoconfig.sumocfg'
        sumo_config_xml.save(sumoconfig_path)
        return sumoconfig_path

    def generate_routes(self, sumo_network: SUMONetwork) -> str:
        sumo_trips = SUMOTrips(sumo_network)
        routes_path = sumo_trips.run_duarouter()
        return routes_path

    def load_modified_network(self):
        sumo_network_modified_path = os.path.join(self.run_path, self.full_run_name + '.json')
        return SUMONetwork.parse_file(sumo_network_modified_path)

    def load_raw_metrics(self, metrics: Optional[List[str]] = None, drop_empty: bool = True) -> pd.DataFrame:
        aggregated_info_file_path = os.path.join(self.run_path, self.full_run_name + '_aggregated_info.xml')
        raw_metrics_df = load_raw_metrics(aggregated_info_file_path, metrics=metrics, drop_empty=drop_empty)
        if len(raw_metrics_df) == 0:
            agg_tripinfo_full_path = os.path.join(self.run_path, 'agg_tripinfo_full.txt')
            metric = 'timeLoss'  # default
            raw_trip_metrics_df = load_raw_trip_metrics_from_attribute_stats(agg_tripinfo_full_path, metric)

            routes_path = os.path.join(self.run_path, self.full_run_name + '_routes.rou.xml')
            routes_xml = RoutesXML(routes_path)
            routes = routes_xml.get_routes()
            routes_df = pd.DataFrame(routes)
            routes_df['id'] = routes_df['id'].apply(lambda x: 'edge' + x)

            # Expand the 'edges' string into a list and explode it to multiple rows
            expanded_routes_df = routes_df.assign(edges=routes_df['edges'].str.split()).explode('edges')

            # Merge with raw_metrics_df to get timeLoss for each edge
            merged_df = expanded_routes_df.merge(raw_trip_metrics_df, left_on='id', right_index=True, how='left')

            raw_metrics_df = merged_df.groupby('edges').agg({'timeLoss': 'sum'})
            raw_metrics_df.index.name = None

        return raw_metrics_df

    def load_metrics(self) -> ExperimentMetrics:
        sumo_network_modified = self.load_modified_network()
        sumo_trips = SUMOTrips(sumo_network_modified)
        # Do we need raw metrics or aggregated over all edges? Let's start with aggregated
        raw_metrics = self.load_raw_metrics()

        # TODO: add main metric selection to the run config
        return ExperimentMetrics.make(raw_metrics, sumo_trips)

    def log(self, *args):
        # logging.info(*args)
        print(*args)

    def run(self):
        """
        0) select what changes in .nod or .edge files we want to do
        1) generate net and routes if needed (netconvert + duarouter)
            1.1) make SUMONetwork file or path the ready one as argument 'base_network_path'
            @see make_base_stockport_net() in network.py

            1.2) save SUMONetwork file path to base_network_path, read it to the variable sumo_network
            1.3) run sumo_network.generate_net_file()

        2) generate trips
        3) generate routes
        4) generate sumoconfigs
        5) run sumo
        6) join results to .ifx file
        7) [optional] plot sumo_traffic_nets, if enabled in config
        """
        skip_all_but_run = True
        sumoconfig_path = os.path.join(self.run_path, self.full_run_name + '_') + 'sumoconfig.sumocfg'
        if not os.path.exists(sumoconfig_path) or not skip_all_but_run:
            os.makedirs(self.run_path, exist_ok=self.config.get('exist_ok', False))

            self.log(f'read sumonetwork at path: {self.base_network_path}')
            sumo_network = SUMONetwork.parse_file(self.base_network_path)
            if sumo_network:
                self.log('success')
            self.log(f'modify the network according to config')
            sumo_network_modified = sumo_network.copy(deep=True)
            sumo_network_modified.dir_path = self.run_path
            sumo_network_modified.name = self.full_run_name
            # TODO: modify network here according to config
            if 'num_lanes' in self.config:
                num_lanes = int(self.config['num_lanes'])
                if 0 < num_lanes < 10:
                    for edge in sumo_network_modified.edges:
                        edge.num_lanes = num_lanes

            sumo_network_modified.save()

            self.log('start generating net file')
            sumo_network_modified.generate_net_file()
            if os.path.exists(sumo_network_modified.make_net_path()):
                self.log('success')
            self.log('start generating routes')
            routes_path = self.generate_routes(sumo_network_modified)
            # sumo_trips = SUMOTrips(sumo_network_modified)
            if os.path.exists(routes_path):
                self.log('success')

            self.log('start generating sumoconfigs')
            sumoconfig_path = self.generate_sumoconfigs(sumo_network_modified)
        if os.path.exists(sumoconfig_path):
            self.log('success')
        self.log('start sumoclient')
        sumo_client = SUMOClient(sumoconfig_path)
        sumo_client.run()
        self.log('Success!')
        # todo save metrics to ifx (or another format), join it to one table, need to be aggregated
        self.metrics = self.load_metrics()
        # TODO: join results to .ifx file


class OptimiserDumpsManager(BaseModel, extra=Extra.allow):
    experiments: List[RegionExperiment]
    current_top_experiments: List[RegionExperiment] = []
    _experiment_hashes: Set[str] = set()
    max_optimisation_cycles: int = 5
    top_n_experiments_to_save: int = 3
    metric_name: str = EDGE_AVG_TRAVELTIME

    def load_hashes(self):
        if len(self.experiments) != len(self._experiment_hashes):
            self._experiment_hashes = set()
            for experiment in self.experiments:
                experiment_config_hash = experiment.make_config_hash()
                self._experiment_hashes.add(experiment_config_hash)
        else:
            self._experiment_hashes = set(self._experiment_hashes)

    def save(self, path: str):
        with open(path, 'w+') as f:
            f.write(self.json())

    def run_all(self, force: bool = False):
        """
        parallel work could be done here
        @param force: bool, run even experiments with calculated metrics
        """
        for experiment in self.experiments:
            if force or experiment.metrics is None:
                experiment.run()

    def has_same_experiment(self, experiment: RegionExperiment) -> bool:
        experiment_config_hash = experiment.make_config_hash()
        return experiment_config_hash in self._experiment_hashes

    def add_experiment(self, experiment: RegionExperiment) -> bool:
        self.load_hashes()
        if not self.has_same_experiment(experiment):
            self.experiments.append(experiment)
            experiment_config_hash = experiment.make_config_hash()
            self._experiment_hashes.add(experiment_config_hash)
            return True
        return False

    def run_optimisation(self, base_model_experiment: RegionExperiment):
        """
        How should optimisation cycle look like?

        # extra idea#1 - maintain several top approaches: by metric, by metric with constrained cost, etc
        # idea#2 - save some random polygons to avoid making tunnels intersecting that polygons (historical center, bad geo?)
        # idea#3 - before simulation, some options could be marked as bad, if they do not change route pathes significantly

        # why batches? there is no parallelization now, but it could be in the future
        # potential changes:
        #   tunnels between which points? use edge shape z-coord to specify tunnel depth
        #

        """
        def update_top_experiments(
                cur_top_experiments: List[RegionExperiment],
                experiments_batch: List[RegionExperiment]
        ):
            cur_top_experiments += experiments_batch  # merge lists
            cur_top_experiments = sorted(cur_top_experiments, key=lambda exp: exp.metrics.get(self.metric_name))
            return cur_top_experiments[:self.top_n_experiments_to_save]

        def get_next_configurations_batch() -> List[RegionExperiment]:
            result_batch = []
            for experiment in self.current_top_experiments:
                upgraded_experiment = self.try_to_upgrade(experiment)
                if not self.has_same_experiment(upgraded_experiment):
                    # make unique run name here
                    upgraded_experiment.run_name = self.make_new_run_name()
                    result_batch.append(upgraded_experiment)
            return result_batch

        if not self.current_top_experiments:
            # base_model_experiment - experiment without extra changes of the real-world model
            self.add_experiment(base_model_experiment)  # it is ok even if the same model already exists in the exp list
            self.current_top_experiments = [base_model_experiment]
        if base_model_experiment.metrics is None:
            base_model_experiment.run()
        k = 0
        logging.info(f'start optimisation for base model: {base_model_experiment.json()}')
        while k < self.max_optimisation_cycles:
            logging.info(f'optimisation loop #{k}')
            experiments_batch = get_next_configurations_batch()
            for experiment in experiments_batch:
                if self.add_experiment(experiment):  # we had no similar experiment in the past
                    logging.info(f'run experiment: {experiment.json()}')
                    experiment.run()
                else:
                    raise Exception('somehow we generated same experiment twice')

            self.current_top_experiments = update_top_experiments(self.current_top_experiments, experiments_batch)
            top_exp = self.current_top_experiments[0]
            print(f'Best metric at loop#{k}: {self.metric_name} = {top_exp.metrics.get(self.metric_name)}')
            print(f'Best metric experiment {top_exp.run_name}: {top_exp.config}')
            k += 1

    def try_to_upgrade(self, experiment) -> RegionExperiment:
        experiment = experiment.copy(deep=True)
        # TODO: add here tunnels, bus routes, etc
        # current_num_lanes = experiment.config.get('num_lanes', 2)
        # experiment.config['num_lanes'] = current_num_lanes + 1
        return experiment

    def find_new_tunnel_location(self):
        """
        TODO: from tunnels.py
        """
        return None

    def make_new_run_name(self) -> str:
        last_run_index = 0
        prefix = 'run'
        prefix_len = len(prefix)
        for exp in self.experiments:
            if exp.run_name.startswith(prefix):
                exp_run_index = 0
                try:
                    exp_run_index = int(exp.run_name[prefix_len:])
                except ValueError:
                    pass
                last_run_index = max(last_run_index, exp_run_index)
        last_run_index += 1
        return f'{prefix}{last_run_index}'


if __name__ == '__main__':

    network_name = 'stockport_base_model'
    base_network_path = os.path.join('gmdata.nosync/', network_name)
    exp_dir_path = os.path.join('gmdata.nosync/', 'stockport_experiments')
    experiment = RegionExperiment(location_name=network_name,
                                  run_name='run0',
                                  dir_path=exp_dir_path,
                                  base_network_path=base_network_path,
                                  config={
                                      'exist_ok': True
                                  })

    manager = OptimiserDumpsManager(experiments=[experiment])
    manager_config_path = '../data.nosync/experiments_config.json'
    manager.save(manager_config_path)
    loaded_manager = OptimiserDumpsManager.parse_file(manager_config_path)
    assert manager == loaded_manager
