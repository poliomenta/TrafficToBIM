#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from bim.sumo.network import make_base_stockport_net, SUMONetwork
from bim.traffic.optimiser import RegionExperiment, OptimiserDumpsManager, make_experiment, EXTRA_TUNNELS_METRIC_PENALTY
from bim.traffic.trips import SUMOTrips
from bim.traffic.metrics import EDGE_TOTAL_TRAVELTIME
from bim.traffic.tunnels import TunnelGenerator, AXIS_LIMITS_CITY_OF_MANCHESTER
import os
import logging
import numpy as np

logging.basicConfig(filename='princess_rd_experiments.log', encoding='utf-8', level=logging.ERROR)

manager_config_path = 'gmdata.nosync/princess_rd_pt_net_experiments_config.json'
manager = None
if os.path.exists(manager_config_path): 
    manager = OptimiserDumpsManager.parse_file(manager_config_path)
[(exp.base_network_path,
    exp.run_name, exp.unique_id, exp.config, exp.metrics, exp.parent_experiment_id,) for exp in manager.experiments]

# manager.experiments[0].metrics = manager.experiments[0].load_metrics()
# manager.reload_all_metrics()


import matplotlib.pyplot as plt
from collections import defaultdict

def plot_metrics(metrics_list, x_func, x_name='reduce_cars_by_percent', 
                 title='Personal-to-public transport transition experiments'):
    metric_values = defaultdict(dict)
    avg_persons_in_car = 1.5
    initial_routes_number = 28242
    
    for exp in manager.experiments:
        reduce_cars_by_percent = exp.config.get('reduce_cars_by_percent', 0)
        people_in_bus_service = 0
        if reduce_cars_by_percent > 0 and exp.metrics.total_bus_trips_count:
            people_in_buses = reduce_cars_by_percent * initial_routes_number * avg_persons_in_car
            people_in_bus_service = people_in_buses / exp.metrics.total_bus_trips_count
        
        x_value = x_func(exp)
        for metric in metrics_list:
            metric_value = exp.metrics.get(metric)
            if 'bus' in metric and metric_value is not None:
                metric_value *= people_in_bus_service
            metric_values[metric][x_value] = metric_value
            if x_value not in metric_values['metrics_sum']:
                metric_values['metrics_sum'][x_value] = metric_value
            elif metric_value is not None:
                metric_values['metrics_sum'][x_value] += metric_value
    exp0 = manager.experiments[0]
    if len(metrics_list) > 1:
        metrics_list = metrics_list + ['metrics_sum']
    fig, ax = plt.subplots()
    for metric in metrics_list:
        l = list(sorted(metric_values[metric].items(), key=lambda x: x[0]))
        keys, values = zip(*l)
        
        plt.plot(keys, values, label=metric)
        if metric in exp0.metrics.dict():
            plt.hlines(exp0.metrics.get(metric), 0, 1, color='red')
        plt.xlabel(metrics_list[0])
    plt.title('Personal-to-public transport transition experiments')
    plt.ylabel(', '.join(metrics_list))
    plt.legend()
    ax.set_ylim(0)
    plt.show()
    
x_func = lambda exp: exp.config.get('reduce_cars_by_percent', 0)
plot_metrics(['total_traveltime', 'total_bus_traveltime'], x_func)
plot_metrics(['total_timeloss', 'total_bus_timeloss'], x_func)
plot_metrics(['avg_traveltime', 'avg_bus_traveltime'], x_func)
plot_metrics(['avg_traveltime'], x_func)
