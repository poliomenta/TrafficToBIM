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
            StockportCenter_run_0_additional_bus_config.xml
            # after sumo run
            StockportCenter_run_0_trip_info.tripinfo.xml
            StockportCenter_run_0_aggregated_info.xml
            ?StockportCenter_run_0_netstate_dump.xml

            !StockportCenter_run_0_model.ifc # includes everything what we need
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
            StockportCenter_run_1_additional_bus_config.xml
"""
import os
from typing import List, Optional, Set

import numpy as np
from pydantic import BaseModel, Extra

from ..traffic.metrics import load_raw_metrics, ExperimentMetrics, EDGE_AVG_TRAVELTIME, \
    load_raw_trip_metrics_from_attribute_stats
from ..traffic.trips import SUMOTrips, RoutesXML
from ..sumo.client import SUMOClient
from ..sumo.network import SUMONetwork
from lxml import etree
import logging
import pandas as pd
import shutil
import uuid

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
    class Config:
        extra = "ignore"  # This will ignore any extra fields not defined in the model

    unique_id: str
    location_name: str
    run_name: str
    dir_path: str
    base_network_path: str  # path to SUMONetwork json file
    base_traffic_net_path: str  # unused
    config: Optional[dict]
    metrics: Optional[ExperimentMetrics] = None
    parent_experiment_id: Optional[str] = None

    def make_config_hash(self):
        return str(hash((self.base_network_path, str(self.config))))

    def calc_tunnels_count(self):
        tunnels = 0
        if 'old_tunnels' in self.config:
            tunnels += len(self.config.get('old_tunnels', []))
        if 'added_tunnels' in self.config:
            tunnels += int(self.config.get('max_add_tunnels', 1))
        return tunnels

    @property
    def run_path(self):
        return os.path.join(self.dir_path, self.run_name)

    @property
    def full_run_name(self):
        return self.location_name + '_' + self.run_name

    def generate_sumoconfigs(self, sumo_network: SUMONetwork) -> str:
        additional_config_path = sumo_network.file_prefix + 'additional_config.xml'
        aggregated_info_path = sumo_network.name + '_aggregated_info.xml'
        aggregated_bus_info_path = sumo_network.name + '_aggregated_bus_info.xml'
        aggregated_info_xml = f"""
<additional>
    <edgeData id="aggregate" vTypes="DEFAULT_VEHTYPE" file="{aggregated_info_path}"/>
    <edgeData id="aggregate" vTypes="BUS" file="{aggregated_bus_info_path}"/>
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

    def generate_routes(self, sumo_network: SUMONetwork, reduce_cars_by_percent: float = 0) -> str:
        sumo_trips = SUMOTrips(sumo_network)
        routes_path = sumo_trips.run_duarouter(reduce_cars_by_percent, force=reduce_cars_by_percent > 0)
        return routes_path

    def load_modified_network(self):
        sumo_network_modified_path = os.path.join(self.run_path, self.full_run_name + '.json')
        print(f'load sumo network from "{sumo_network_modified_path}"')
        return SUMONetwork.parse_file(sumo_network_modified_path)

    def load_raw_metrics(self, metrics: Optional[List[str]] = None, drop_empty: bool = True):
        aggregated_info_file_path = os.path.join(self.run_path, self.full_run_name + '_aggregated_info.xml')
        aggregated_bus_info_file_path = os.path.join(self.run_path, self.full_run_name + '_aggregated_bus_info.xml')
        raw_metrics_df, raw_bus_metrics_df = load_raw_metrics(aggregated_info_file_path, aggregated_bus_info_file_path,
                                                              metrics=metrics, drop_empty=drop_empty)
        return raw_metrics_df, raw_bus_metrics_df

    def load_metrics(self) -> ExperimentMetrics:
        sumo_network_modified = self.load_modified_network()
        sumo_trips = SUMOTrips(sumo_network_modified)
        # Do we need raw metrics or aggregated over all edges? Let's start with aggregated
        raw_metrics, raw_bus_metrics_df = self.load_raw_metrics()

        # need to add main metric selection to the run config
        return ExperimentMetrics.make(raw_metrics, raw_bus_metrics_df, sumo_trips)

    def log(self, *args):
        # logging.info(*args)
        print(*args)

    def fork(self):
        forked_exp = self.copy(deep=True)
        forked_exp.unique_id = generate_uuid()
        forked_exp.parent_experiment_id = self.unique_id

        added_tunnels = self.config.get('added_tunnels', None)
        if added_tunnels is not None:
            self.config['old_tunnels'] = self.config.get('old_tunnels', []) + added_tunnels
            del self.config['added_tunnels']
        return forked_exp

    def run(self, exp_manager: 'OptimiserDumpsManager'):
        try:
            self.run_impl(exp_manager)
        except Exception as e:
            print(e)
            self.metrics = ExperimentMetrics.make_failed()

    def run_impl(self, exp_manager: 'OptimiserDumpsManager'):
        skip_all_but_run = True
        sumoconfig_path = os.path.join(self.run_path, self.full_run_name + '_') + 'sumoconfig.sumocfg'
        if not os.path.exists(sumoconfig_path) or not skip_all_but_run:
            os.makedirs(self.run_path, exist_ok=self.config.get('exist_ok', False))

            parent_experiment = exp_manager.get_experiment_by_guid(self.parent_experiment_id)
            if self.parent_experiment_id and parent_experiment is None:
                raise Exception('parent experiment is not found')
            if self.parent_experiment_id:
                sumo_network = parent_experiment.load_modified_network()
            else:
                self.log(f'read sumonetwork at path: {self.base_network_path}')
                sumo_network = SUMONetwork.parse_file(self.base_network_path)

            if sumo_network:
                self.log('success')
            if parent_experiment is not None and self.config.get('added_tunnels', None) is not None:
                # it updates 'added_tunnels' config field
                tunnel_generator, new_config = add_random_tunnels_to_the_net(parent_experiment, self)
                sumo_network_modified = tunnel_generator.sumo_network
                self.config = new_config
            else:
                self.log(f'modify the network according to config')
                sumo_network_modified = sumo_network.copy(deep=True)
                # add modification of the road network here, except for the addition of tunnels

            if self.config.get('enable_bus_trips_and_lanes', False):
                sumo_network_modified.enable_bus_trips_and_lanes = True
                for edge in sumo_network_modified.edges:
                    if 'tunnel' not in edge.edge_id.id:  # let's make tunnels for cars only
                        edge.bus_lane = True

            sumo_network_modified.dir_path = self.run_path
            sumo_network_modified.name = self.full_run_name
            sumo_network_modified.save()

            self.log('start generating net file')
            sumo_network_modified.generate_net_file()
            if os.path.exists(sumo_network_modified.make_net_path()):
                self.log('success')

            trips_path = SUMOTrips(sumo_network).trips_path
            if os.path.exists(trips_path):
                new_trips_path = SUMOTrips(sumo_network_modified).trips_path
                print(f'Copy {trips_path} to {new_trips_path}')
                shutil.copy(trips_path, new_trips_path)
            self.log('start generating routes')
            reduce_cars_by_percent = self.config.get('reduce_cars_by_percent', 0)
            routes_path = self.generate_routes(sumo_network_modified, reduce_cars_by_percent)
            # sumo_trips = SUMOTrips(sumo_network_modified)
            if os.path.exists(routes_path):
                self.log('success')

            self.log('start generating sumoconfigs')
            sumoconfig_path = self.generate_sumoconfigs(sumo_network_modified)

        added_tunnels = self.config.get('added_tunnels', None)
        if added_tunnels is not None:
            self.config['old_tunnels'] = self.config.get('old_tunnels', []) + added_tunnels
            del self.config['added_tunnels']

        if os.path.exists(sumoconfig_path):
            self.log('success')
        self.log('start sumoclient')
        sumo_client = SUMOClient(sumoconfig_path)
        sumo_client.run()
        self.log('Success!')
        # need to save metrics to ifc (or another format), join it to one table, need to be aggregated
        self.metrics = self.load_metrics()
        # need to  join results to .ifc file

    def make_random_seed_from_run_name(self):
        import hashlib
        run_name_b = self.run_name.encode('utf-8')
        hash_bytes = hashlib.md5(run_name_b).digest()
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        random_seed = hash_int % (2 ** 32)
        return random_seed


def generate_uuid() -> str:
    return uuid.uuid4().hex


def make_experiment(**args):
    unique_id = generate_uuid()
    return RegionExperiment(unique_id=unique_id, **args)


EXTRA_TUNNELS_METRIC_PENALTY = 0.2  # the coefficient should be the same order of magnitude as metric


class OptimiserDumpsManager(BaseModel, extra=Extra.allow):
    experiments: List[RegionExperiment]
    current_top_experiments: List[RegionExperiment] = []
    _experiment_hashes: Set[str] = set()
    max_optimisation_cycles: int = 5
    top_n_experiments_to_save: int = 3
    metric_name: str = EDGE_AVG_TRAVELTIME

    def load_hashes(self):
        self._experiment_hashes = set()
        for experiment in self.experiments:
            experiment_config_hash = experiment.make_config_hash()
            self._experiment_hashes.add(experiment_config_hash)

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
                experiment.run(self)

    def has_same_experiment(self, experiment: RegionExperiment) -> bool:
        experiment_config_hash = experiment.make_config_hash()
        return experiment_config_hash in self._experiment_hashes

    def add_experiment(self, experiment: RegionExperiment) -> bool:
        self.load_hashes()
        if not self.has_same_experiment(experiment):
            # make unique run name here
            experiment.run_name = self.make_new_run_name()
            self.experiments.append(experiment)
            experiment_config_hash = experiment.make_config_hash()
            self._experiment_hashes.add(experiment_config_hash)
            return True
        return False

    def run_optimisation(self, base_model_experiment: RegionExperiment, save_path: str):
        """
        # extra idea#1 - maintain several top approaches: by metric, by metric with constrained cost, etc
        # idea#2 - save some random polygons to avoid making tunnels intersecting that polygons (historical center, bad geo?)
        # idea#3 - before simulation, some options could be marked as bad, if they do not change route pathes significantly

        # why use batches? there is no parallelization now, but it could be in the future
        """
        def update_top_experiments(
                cur_top_experiments: List[RegionExperiment],
                experiments_batch: List[RegionExperiment]
        ):
            def get_exp_metric(exp):
                if exp.metrics is not None:
                    metric_value = exp.metrics.get(self.metric_name)
                    return metric_value + exp.calc_tunnels_count() * EXTRA_TUNNELS_METRIC_PENALTY
                failed_experiment_metric_value = float('inf')
                return failed_experiment_metric_value
            cur_top_experiments += experiments_batch  # merge lists
            cur_top_experiments = sorted(cur_top_experiments, key=get_exp_metric)
            return cur_top_experiments[:self.top_n_experiments_to_save]

        def get_next_configurations_batch(base_model_experiment) -> List[RegionExperiment]:
            self.load_hashes()
            result_batch = []
            parent_experiments = self.current_top_experiments
            if base_model_experiment not in parent_experiments:
                parent_experiments.append(base_model_experiment)
            for experiment in parent_experiments:
                upgraded_experiments = self.try_to_upgrade(experiment)
                for upgraded_experiment in upgraded_experiments:
                    if not self.has_same_experiment(upgraded_experiment):
                        result_batch.append(upgraded_experiment)
            np.random.shuffle(result_batch)
            return result_batch

        if not self.current_top_experiments:
            # base_model_experiment - experiment without extra changes of the real-world model
            self.add_experiment(base_model_experiment)  # it is ok even if the same model already exists in the exp list
            self.current_top_experiments = [base_model_experiment]
        if base_model_experiment.metrics is None:
            base_model_experiment.run(self)
        k = 0
        logging.info(f'start optimisation for base model: {base_model_experiment.json()}')
        while k < self.max_optimisation_cycles:
            logging.info(f'optimisation loop #{k}')
            experiments_batch = get_next_configurations_batch(base_model_experiment)
            if len(experiments_batch) == 0:
                print('Empty experiments_batch')
            for experiment in experiments_batch:
                if self.add_experiment(experiment):  # we had no similar experiment in the past
                    print(f'run experiment: {experiment.json()}')
                    experiment.run(self)
                    self.save(save_path)
                else:
                    cur_exp_hash = experiment.make_config_hash()
                    same_experiment = list(filter(lambda x: x.make_config_hash() == cur_exp_hash, self.experiments))[0]
                    raise Exception(f'somehow we generated same experiment twice {same_experiment.run_name}')

            self.current_top_experiments = update_top_experiments(self.current_top_experiments, experiments_batch)
            top_exp = self.current_top_experiments[0]
            print(f'Best metric at loop#{k}: {self.metric_name} = {top_exp.metrics.get(self.metric_name)}')
            print(f'Best metric experiment {top_exp.run_name}: {top_exp.config}')
            k += 1
            self.save(save_path)

    def try_to_upgrade(self, parent_experiment: RegionExperiment) -> List[RegionExperiment]:
        if parent_experiment.config.get('try_reduce_cars', False):
            result = []
            for enable_bus_trips in [True]:  # add False here to simulate pure car decreasing effect, without buses
                for perc in np.arange(0.02, 1.02, 0.02):
                    if enable_bus_trips:
                        result.append(self.make_new_exp_reduce_cars_by_percent_and_add_bus_trips(parent_experiment, perc))
                    else:
                        result.append(self.make_new_exp_reduce_cars_by_percent(parent_experiment, perc))
            return result

        experiment = parent_experiment.fork()
        # add here tunnels generation config, bus routes, etc
        experiment.config['max_add_tunnels'] = 1
        experiment.config['n_random_path'] = 100
        experiment.config['added_tunnels'] = []  # acts as flag that we need to generate new tunnels, fill it later

        return [experiment]

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

    def get_experiment_by_guid(self, parent_experiment_id: str):
        for exp in self.experiments:
            if exp.unique_id == parent_experiment_id:
                return exp
        return None

    def make_new_exp_reduce_cars_by_percent(self, parent_experiment: RegionExperiment, percent: float = 0.1):
        experiment = parent_experiment.fork()
        experiment.config['reduce_cars_by_percent'] = percent
        experiment.run_name = self.make_new_run_name()
        return experiment

    def make_new_exp_reduce_cars_by_percent_and_add_bus_trips(self, parent_experiment: RegionExperiment, percent: float = 0.1):
        experiment = parent_experiment.fork()
        experiment.config['enable_bus_trips_and_lanes'] = True
        experiment.config['reduce_cars_by_percent'] = percent
        experiment.run_name = self.make_new_run_name()
        return experiment

    def reload_all_metrics(self):
        for exp in self.experiments:
            exp.metrics = exp.load_metrics()

def add_random_tunnels_to_the_net(parent_exp: RegionExperiment, exp: RegionExperiment, verbose: bool = False):
    from .tunnels import AXIS_LIMITS_MANCHESTER_SOUTH, TunnelGenerator
    import warnings
    warnings.filterwarnings('ignore')
    axis_limits = exp.config.get('axis_limits', AXIS_LIMITS_MANCHESTER_SOUTH)
    percent_timeloss = exp.config.get('percent_timeloss', 0.98)
    min_tunnel_len = exp.config.get('min_tunnel_len', 500)
    n_random_path = exp.config.get('n_random_path', 20)
    path_edge_count = exp.config.get('path_edge_count', 20)
    max_add_tunnels = exp.config.get('max_add_tunnels', 1)
    max_generate_tunnel_attempts = exp.config.get('max_generate_tunnel_attempts', 20)
    random_seed = exp.config.get('random_seed', None)
    if random_seed is None:
        random_seed = exp.make_random_seed_from_run_name()
    np.random.seed(random_seed)
    disable_tqdm = not verbose
    good_random_paths = []
    tg = None
    k = 0
    while len(good_random_paths) == 0:
        if k > 0:
            print(f'Attempt to generate tunnels again: {k}')
        tg = TunnelGenerator(parent_exp, axis_limits=axis_limits, percent_timeloss=percent_timeloss, min_tunnel_len=min_tunnel_len,
                             path_edge_count=path_edge_count, disable_tqdm=disable_tqdm)
        graph_id_to_nx_graph = tg.make_nx_graphs(tg.graph_components_df)
        graph_id_to_paths = tg.generate_random_paths(graph_id_to_nx_graph, n_random_path, verbose)
        good_random_paths = [path for path_list in graph_id_to_paths.values() for path in path_list]
        k += 1
        if k >= max_generate_tunnel_attempts:
            raise Exception(f'no valid tunnel found for experiment {exp.run_name}')

    np.random.shuffle(good_random_paths)
    exp.config['added_tunnels'] = tg.add_tunnels_to_sumo_network(good_random_paths[:max_add_tunnels], verbose)
    warnings.filterwarnings('default')
    return tg, exp.config


def test_princess_rd_base_net_experiment_run():
    manager_config_path = 'gmdata.nosync/princess_rd_pt_net_experiments_config.json'
    manager = None
    if os.path.exists(manager_config_path):
        manager = OptimiserDumpsManager.parse_file(manager_config_path)
    if not manager:
        network_name = 'princess_rd_pt_net'
        base_traffic_net_path = os.path.join('gmdata.nosync/', network_name)
        base_network_path = os.path.join(base_traffic_net_path, f'{network_name}.json')
        exp_dir_path = os.path.join('gmdata.nosync/', 'princess_rd_pt_experiments/')
        experiment = make_experiment(location_name='princess_rd',
                                     run_name='run0',
                                     dir_path=exp_dir_path,
                                     base_network_path=base_network_path,
                                     base_traffic_net_path=base_traffic_net_path,
                                     config={
                                         'exist_ok': True,
                                         'try_reduce_cars': True
                                     })
        manager = OptimiserDumpsManager(experiments=[experiment], max_optimisation_cycles=100,
                                        top_n_experiments_to_save=3)
        manager.save(manager_config_path)
        manager.experiments[0].run(manager)
        manager.save(manager_config_path)

    manager.run_optimisation(manager.experiments[0], manager_config_path)
    metrics_df = pd.DataFrame([
        {
            **experiment.metrics.dict(),
            **experiment.dict(),
        }
        for experiment in manager.experiments])
    pd.options.display.max_columns = None
    metrics_df.to_csv(manager_config_path)
    print(metrics_df)
    manager.save(manager_config_path)
    return manager

if __name__ == '__main__':
    test_princess_rd_base_net_experiment_run()