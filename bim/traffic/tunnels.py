"""
1) read *aggregated_info.xml to get edge table data
2) add coordinates from *edge.edg.xml to be able plot these edges
3) select adjacent edges with high timeLoss value and make a graph from them
4) generate potential tunnel routes for the graph

"""
from .optimiser import RegionExperiment
import numpy as np

from collections import defaultdict
from heapq import heappush, heappop
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.ops import substring
from shapely import affinity
from shapely.geometry import Point, LineString
import warnings


CONGESTED_STREET = 1
CONGESTED_NEIGHBOUR_STREET = 2

DEFAULT_FIGURE_SIZE = (15, 10)


def plot_path(gdf):
    def f(ax):
        gdf['shape'].plot(ax=ax)
    return f


class GraphUtils:
    @staticmethod
    def make_graph_from_pairs(pairs_df):
        edges = defaultdict(set)
        for i in range(len(pairs_df)):
            row = pairs_df.iloc[i]
            street_from, street_to = row['edge_id_x'], row['edge_id_y']
            edges[street_from].add(street_to)
            edges[street_to].add(street_from)
        return edges

    @staticmethod
    def get_street_neighbours(streets_df, max_distance):
        street_spatial_index = streets_df.sindex

        def f(street):
            street_geometry = street['shape']
            possible_matches_index = list(street_spatial_index.nearest(street_geometry))
            possible_matches_index = np.asarray(possible_matches_index)[1, :]
            possible_matches_index = list(set(possible_matches_index))
            possible_matches = streets_df.iloc[possible_matches_index]
            precise_matches = possible_matches[(possible_matches.distance(street_geometry) <= max_distance)
                                               & (possible_matches.edge_id != street.edge_id)]
            return precise_matches

        return f

    @staticmethod
    def find_dijkstra_path(streets_df, max_distance):
        get_street_neighbours_f = GraphUtils.get_street_neighbours(streets_df, max_distance)

        def f(street_from_index, street_to_index):
            street_to = streets_df.iloc[street_to_index]
            heap = [(0, street_from_index, [])]
            visited = set()
            while heap:
                distance, street_index, path = heappop(heap)
                if street_index == street_to_index:
                    return path + [street_to_index]
                path = path + [street_index]

                street = streets_df.iloc[street_index]
                if street.edge_id in visited:
                    continue
                visited.add(street.edge_id)
                street_neighbours_df = get_street_neighbours_f(street)

                for i in range(len(street_neighbours_df)):
                    neighb_street = street_neighbours_df.iloc[i]
                    if neighb_street.edge_id not in visited:
                        neighb_street_index = neighb_street.name
                        heappush(heap, (neighb_street['shape'].distance(street_to['shape']), neighb_street_index, path))
            return None

        return f

    @staticmethod
    def get_graph_components(pairs_df):
        edges = GraphUtils.make_graph_from_pairs(pairs_df)

        def dfs(node):
            graph_components[node] = graph_components_id
            for adj_node in edges[node]:
                if adj_node not in graph_components:
                    dfs(adj_node)

        graph_components_id = 0
        graph_components = {}
        for i in range(len(pairs_df)):
            row = pairs_df.iloc[i]
            street_from = row['edge_id_x']
            if street_from not in graph_components:
                dfs(street_from)
                graph_components_id += 1
        return pd.DataFrame(list(graph_components.items()), columns=['edge_id', 'graph_id'])


    @staticmethod
    def generate_random_optimal_paths(G, sample_size, path_length=5, weight="weight"):
        """
        Adjusted original nx.generate_random_paths function to make paths without repeating nodes and without sharp turns
        https://github.com/networkx/networkx/blob/49d1e13b7cb9e31595f9abcff70c7fdd3932897f/networkx/algorithms/similarity.py#L1610
        """
        import numpy as np

        adj_mat = nx.to_numpy_array(G, weight=weight)

        node_map = np.array(G)
        num_nodes = G.number_of_nodes()
        visited_nodes = np.zeros((num_nodes,))

        for path_index in range(sample_size):
            visited_nodes[:] = 0
            node_index = np.random.randint(0, high=num_nodes)
            node = node_map[node_index]
            visited_nodes[node_index] = 1

            path = [node]

            starting_index = node_index
            for _ in range(path_length):
                p_list = np.clip(adj_mat[starting_index] - visited_nodes, 0, 1000)
                if sum(p_list) <= 0:
                    break
                neighbor_index = np.random.choice(
                    num_nodes, p=p_list / sum(p_list)
                )
                visited_nodes[neighbor_index] += 1
                starting_index = neighbor_index
                neighbor_node = node_map[neighbor_index]
                path.append(neighbor_node)
            yield path


class VectorUtils:
    @staticmethod
    def angle_between_vectors(v1, v2):
        """Returns the angle in degrees between vectors 'v1' and 'v2'."""
        if (v1 == v2).all() or (v1 == -v2).all():
            return 0

        dot_product = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        angle = np.degrees(np.arccos(dot_product / (v1_norm * v2_norm)))

        return angle

    @staticmethod
    def find_sharp_turns(gdf, threshold=90, f=None):
        """
        Detect sharp turns greater than the specified threshold between consecutive streets in the GeoDataFrame.
        Returns a list of indexes where sharp turns were detected.
        """
        sharp_turns = []
        not_sharp_turns = []

        for idx in range(len(gdf) - 1):
            # Current and next LineString
            line1, line2 = gdf.geometry.iloc[idx], gdf.geometry.iloc[idx + 1]

            # Last point of the first LineString and the first point of the next LineString
            # We assume here that the LineStrings are ordered and connected end to start
            p1, p2, p3 = line1.coords[-1], line1.coords[-2], line2.coords[1]

            # Vectors
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p1)

            angle = VectorUtils.angle_between_vectors(v1, v2)

            if f is not None:
                print(idx, angle, p1, p2, p3)
                fig, ax = plt.subplots()
                f(ax)
                gdf.iloc[idx:idx + 1]['shape'].plot(ax=ax)
                gdf.iloc[idx + 1:idx + 2]['shape'].plot(ax=ax, color='red')

                plt.arrow(p1[0], p1[1], v2[0], v2[1], width=5, color='orange')
                plt.arrow(p1[0], p1[1], v1[0], v1[1], width=5)
                plt.text(p1[0], p1[1], 's')
                plt.text(p2[0], p2[1], '1')
                plt.text(p3[0], p3[1], '2')
                plt.show()

            if angle < threshold:
                sharp_turns.append(idx)
            else:
                not_sharp_turns.append(idx)

        return sharp_turns, not_sharp_turns


class TunnelGenerator:
    def __init__(self, exp: RegionExperiment):
        sumo_network = exp.load_modified_network()
        self.sumo_network = sumo_network
        self.nodes_df = sumo_network.make_nodes_df()
        self.edges_df = sumo_network.make_edges_df()
        self.raw_metrics = exp.load_raw_metrics()
        self.raw_metrics['edge_id'] = self.raw_metrics.index
        self.joined_metrics = None
        self.edges_from_to_pairs = None
        self.congested_edge_pairs = None

        self.top2_perc_timeLoss_value = None
        self.graph_components_df = None
        self.edge_to_graph_id = None
        self.graph_ids = None

        self.make_joined_metrics_and_edge_pairs()

        self.make_congested_street_graphs()

    def make_joined_metrics_and_edge_pairs(self):
        self.joined_metrics = self.edges_df.merge(self.raw_metrics, on=['edge_id'], how='left')
        self.joined_metrics = self.mark_congested(self.joined_metrics)

        edges_to = self.joined_metrics[['edge_id', 'from_id', 'length', 'timeLoss', 'congested']]
        edges_from = self.joined_metrics[['edge_id', 'to_id', 'length', 'timeLoss', 'congested']]
        edges_from_to_pairs = edges_from.merge(edges_to, left_on='to_id', right_on='from_id')
        edges_from_to_pairs['total_length'] = edges_from_to_pairs['length_x'] + edges_from_to_pairs['length_y']
        edges_from_to_pairs['total_timeLoss'] = edges_from_to_pairs['timeLoss_x'] + edges_from_to_pairs['timeLoss_y']
        edges_from_to_pairs['min_congested'] = np.min(
            (edges_from_to_pairs['congested_x'], edges_from_to_pairs['congested_y']), axis=0)
        self.edges_from_to_pairs = edges_from_to_pairs

    def make_congested_street_graphs(self):
        congested_edge_pair_index = self.edges_from_to_pairs.apply(lambda x: x['min_congested'] >= 1, axis=1).dropna()
        self.congested_edge_pairs = self.edges_from_to_pairs[congested_edge_pair_index]
        self.graph_components_df = GraphUtils.get_graph_components(self.congested_edge_pairs)
        self.edge_to_graph_id = {row.edge_id: row.graph_id for row in self.graph_components_df.itertuples()}
        self.graph_ids = set(self.graph_components_df.graph_id)

    def mark_congested(self, joined_metrics):
        joined_metrics['congested'] = (joined_metrics['timeLoss'] >= self.top2_perc_timeLoss).astype(np.int32)
        congested_streets_df = joined_metrics[joined_metrics['congested'] == CONGESTED_STREET].reset_index()
        max_distance = 2000
        # find path using all edges
        find_dijkstra_path_f = GraphUtils.find_dijkstra_path(joined_metrics, max_distance)

        n = len(congested_streets_df)

        congested_street_distance_threshold = 200
        for street_from_index in range(n):
            street_from = congested_streets_df.iloc[street_from_index]
            for street_to_index in range(street_from_index + 1, n):
                street_to = congested_streets_df.iloc[street_to_index]
                if street_from['shape'].distance(street_to['shape']) < congested_street_distance_threshold:
                    street_from_original_index = street_from['index']
                    street_to_original_index = street_to['index']
                    for edge_index in find_dijkstra_path_f(street_from_original_index, street_to_original_index):
                        if joined_metrics.iloc[edge_index].congested == CONGESTED_STREET:
                            continue
                        joined_metrics.loc[edge_index, 'congested'] = CONGESTED_NEIGHBOUR_STREET
        return joined_metrics

    @property
    def top2_perc_timeLoss(self):
        if self.top2_perc_timeLoss_value is None:
            filtered_timeLoss = self.joined_metrics['timeLoss'].dropna()
            self.top2_perc_timeLoss_value = sorted(filtered_timeLoss)[int(len(filtered_timeLoss) * 0.98)]
        return self.top2_perc_timeLoss_value

    def plot_graphs(self, cmap):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        self.plot_all_streets(ax)
        graph_ids = self.graph_ids
        color_step = cmap.N // (len(graph_ids) - 1)
        colors = [cmap(i * color_step) for i in range(len(graph_ids))]
        graph_id_to_color = dict(zip(graph_ids, colors))

        top_length_edge_pairs = self.congested_edge_pairs.sort_values(by='total_timeLoss', ascending=False).reset_index(
            drop=True)
        for target_graph_id in graph_ids:
            text_plotted = False
            graph_color = graph_id_to_color[target_graph_id]

            for i in range(len(self.congested_edge_pairs)):
                edge_from = top_length_edge_pairs.loc[i, 'edge_id_x']
                edge_to = top_length_edge_pairs.loc[i, 'edge_id_y']
                if self.edge_to_graph_id[edge_from] != target_graph_id:
                    continue

                selected_edges_df = self.get_edges_slice_by_id([edge_from, edge_to])
                selected_edges_df.plot(color=graph_color, ax=ax)
                coord = selected_edges_df.iloc[0]['shape'].coords[0]
                if not text_plotted:
                    plt.text(coord[0], coord[1] + 500, target_graph_id, color=graph_color)
                    text_plotted = True
        ax.set_xlim(370000, 390000)
        ax.set_ylim(390000, 400000)

        # joined_metrics[joined_metrics['timeLoss'] > top2_perc_timeLoss].plot(ax=ax)
        #     plt.show()

    def get_edges_slice_by_id(self, edge_ids):
        return self.edges_df.set_index('edge_id').loc[edge_ids]

    def calc_path_length(self, edge_ids):
        return self.get_edges_slice_by_id(edge_ids).length.sum()

    def remove_duplicated_nodes(self, gdf):
        has_occurred_before = gdf['to_id'].duplicated(keep='first')
        first_true_index = has_occurred_before.idxmax()
        last_good_point = gdf.index.get_loc(first_true_index) - 1
        return gdf.iloc[:last_good_point]

    def make_nx_graphs(self, graph_components_df):
        graph_ids = set(graph_components_df['graph_id'])
        graph_id_to_nx_graph = {}
        for graph_id in graph_ids:
            G = nx.Graph()
            for node in set(self.congested_edge_pairs['edge_id_x']).union(set(self.congested_edge_pairs['edge_id_y'])):
                if self.edge_to_graph_id[node] == graph_id:
                    G.add_node(node, graph_id=graph_id)
            for i in range(len(self.congested_edge_pairs)):
                row = self.congested_edge_pairs.iloc[i]
                if self.edge_to_graph_id[row.edge_id_x] == graph_id:
                    G.add_edge(row.edge_id_x, row.edge_id_y)
                    G.add_edge(row.edge_id_y, row.edge_id_x)
            graph_id_to_nx_graph[graph_id] = G

        return graph_id_to_nx_graph

    def plot_random_paths(self, graph_id_to_nx_graph, cmap, n_random_path=3, alpha=0.3, min_path_len=1000):
        graph_ids = list(graph_id_to_nx_graph.keys())
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        self.plot_all_streets(ax)
        colors_count = len(graph_ids) * n_random_path
        color_step = cmap.N // (colors_count - 1)
        colors = [cmap(i * color_step) for i in range(colors_count)]

        graph_id_and_path_id_to_color = {}
        color_index = 0
        for graph_id in graph_ids:
            for i in range(n_random_path):
                graph_id_and_path_id_to_color[f'{graph_id}_{i}'] = colors[color_index]
                color_index += 1

        good_random_paths = []
        for target_graph_id in graph_ids:
            print(f'{target_graph_id=}')
            random_paths = GraphUtils.generate_random_optimal_paths(graph_id_to_nx_graph[target_graph_id], n_random_path,
                                                         path_length=20)
            for i, random_path in enumerate(random_paths):
                path_len = self.calc_path_length(random_path)
                if path_len < min_path_len:
                    #                 print(f'skip path with len {path_len}')
                    continue

                candidate_paths = self.preprocess_path_candidates([random_path])
                if len(candidate_paths) == 0:
                    continue
                random_path = candidate_paths[0]

                path_len = self.calc_path_length(random_path)
                if path_len < min_path_len:
                    #                 print(f'skip path with len {path_len}')
                    continue

                print(f'random_path #{i} with len {path_len}')
                good_random_paths.append(random_path)
                text_plotted = False
                graph_color = graph_id_and_path_id_to_color[f'{target_graph_id}_{i}']

                for edge in random_path:
                    selected_edges_df = self.get_edges_slice_by_id([edge])
                    selected_edges_df.plot(color=graph_color, ax=ax, alpha=alpha)
                    coord = selected_edges_df.iloc[0]['shape'].coords[0]
                    if not text_plotted:
                        plt.text(coord[0], coord[1] + 100, f'{target_graph_id}_{i}', color=graph_color)
                        text_plotted = True
        ax.set_xlim(380000, 385000)
        ax.set_ylim(392000, 400000)
        return good_random_paths

    @staticmethod
    def split_id_by_sharp_turns(sharp_turns, not_sharp_turns):
        ids = sorted(list(set(sharp_turns + not_sharp_turns)))
        sharp_turns_set = set(sharp_turns)
        result = ''
        for id in ids:
            if id in sharp_turns_set:
                if result[-1] != '|':
                    result += '|'
            else:
                result += str(id) + ','
        groups = [x.split(',')[:-1] for x in result.split('|')]
        biggest_group = sorted(groups, key=len, reverse=True)[0]
        return list(map(int, biggest_group))

    @staticmethod
    def reverse_shape(ls):
        return substring(ls, ls.length, 0)

    def get_node_coord(self, node_id):
        return self.nodes_df.set_index('node_id').loc[node_id].point

    def get_path_coordinates(self, path):
        path_edge_df = self.get_edges_slice_by_id(path)
        return path_edge_df

    @staticmethod
    def get_last_point(linestring):
        last = Point(linestring.coords[-1])
        return last

    @staticmethod
    def get_first_point(linestring):
        first = Point(linestring.coords[0])
        return first

    def add_shape_first_last_points(self, gdf):
        for i in range(len(gdf) - 1):
            next_i = gdf.index[i + 1]
            i = gdf.index[i]
            if gdf.loc[i, 'to_id'] != gdf.loc[next_i, 'from_id']:
                gdf.loc[i, 'to_id'], gdf.loc[i, 'from_id'] = gdf.loc[i, 'from_id'], gdf.loc[i, 'to_id']
                gdf.loc[i, 'shape'] = self.reverse_shape(gdf.loc[i, 'shape'])

        gdf['first_point'] = gdf['from_id'].apply(self.get_node_coord)
        gdf['last_point'] = gdf['to_id'].apply(self.get_node_coord)
        return gdf

    def preprocess_path_candidates(self, paths, max_angle=90):
        result_paths = []
        angle = 180 - abs(max_angle)
        for i in range(len(paths)):
            gdf = self.add_shape_first_last_points(self.get_path_coordinates(paths[i]))
            gdf = self.remove_duplicated_nodes(gdf)
            sharp_turns, not_sharp_turns = VectorUtils.find_sharp_turns(gdf, threshold=angle)
            biggest_smooth_part = self.split_id_by_sharp_turns(sharp_turns, not_sharp_turns)
            gdf = gdf.iloc[biggest_smooth_part]
            if len(gdf) > 0:
                result_paths.append(gdf.reset_index()['edge_id'].tolist())
        return result_paths

    def plot_tunnel(self, edges, ax, color='black'):
        warnings.filterwarnings('ignore')
        gdf = self.get_path_coordinates(edges)
        gdf = self.add_shape_first_last_points(gdf)
        gdf = self.remove_duplicated_nodes(gdf)
        shift_x, shift_y = -100, -100

        first_line = LineString([gdf.iloc[0].first_point, affinity.translate(gdf.iloc[0].last_point, shift_x, shift_y)])
        last_line = LineString(
            [affinity.translate(gdf.iloc[-1].first_point, shift_x, shift_y), gdf.iloc[-1].last_point])

        gdf['new_shape'] = None
        gdf.loc[1:-1, 'new_shape'] = gdf[1:-1].translate(shift_x, shift_y)

        gdf.loc[0, 'new_shape'] = first_line
        gdf.loc[-1, 'new_shape'] = last_line

        gdf['shape'] = gdf['new_shape']
        gdf['shape'].plot(ax=ax, color=color)
        warnings.filterwarnings('default')
        return gdf

    def plot_good_random_paths(self):
        graph_id_to_nx_graph = self.make_nx_graphs(self.graph_components_df)
        sample_graph_id_to_nx_graph = {key: graph_id_to_nx_graph[key] for key in [1]}
        good_random_paths = self.plot_random_paths(sample_graph_id_to_nx_graph, plt.cm.rainbow, 200, min_path_len=2000)
        return good_random_paths

    def plot_all_streets(self, ax):
        self.joined_metrics.plot(ax=ax, color='lightgray')

    def plot_tunnels(self):
        good_random_paths = self.plot_good_random_paths()
        for i in range(len(good_random_paths)):
            gdf = self.add_shape_first_last_points(self.get_path_coordinates(good_random_paths[i]))
            gdf = self.remove_duplicated_nodes(gdf)
            # f = plot_path(gdf)
            # f = None
            # sharp_turns, not_sharp_turns = VectorUtils.find_sharp_turns(gdf, f=f)

            fig, ax = plt.subplots()
            self.plot_all_streets(ax)
            ax.set_title(f'congested graph{i} with tunnel (black)')
            gdf['shape'].plot(ax=ax)

            # if sharp_turns:
            #     gdf.iloc[sharp_turns]['shape'].plot(ax=ax, color='red')

            self.plot_tunnel(good_random_paths[i], ax)
            ax.set_xlim(380000, 385000)
            ax.set_ylim(392000, 400000)
            plt.plot()
        return good_random_paths

    def plot_intermediate_congested_streets(self):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        self.plot_all_streets(ax)
        self.joined_metrics[self.joined_metrics['congested'] == CONGESTED_NEIGHBOUR_STREET].plot(ax=ax, color='red', alpha=0.5,
                                                                                       linewidth=3)
        self.joined_metrics[self.joined_metrics['congested'] == CONGESTED_STREET].plot(ax=ax, color='black', alpha=0.5,
                                                                             linewidth=3)
        ax.set_xlim(380000, 390000)
        ax.set_ylim(390000, 400000)

    def plot_congested_streets(self):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        self.plot_all_streets(ax)
        self.joined_metrics[self.joined_metrics['timeLoss'] > self.top2_perc_timeLoss].plot(ax=ax)
        ax.set_xlim(380000, 390000)
        ax.set_ylim(390000, 400000)

    @staticmethod
    def get_metric_color(metric_series, cmap):
        min_metric = min(metric_series.dropna())
        max_metric = max(metric_series.dropna())

        def f(value):
            if pd.isna(value):
                return 'grey'
            cmap_value = int(cmap.N * (value - min_metric) / (max_metric - min_metric))
            return cmap(cmap_value)

        return f

    def plot_whole_net_metric(self, metric_name: str, city_of_manch_only: bool = False):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        get_metric_color_f = self.get_metric_color(self.joined_metrics[metric_name], plt.cm.jet)
        self.joined_metrics[f'{metric_name}_color'] = self.joined_metrics[metric_name].apply(get_metric_color_f)
        self.joined_metrics.plot(ax=ax, color=self.joined_metrics[f'{metric_name}_color'])
        if city_of_manch_only:
            ax.set_xlim(380000, 390000)
            ax.set_ylim(390000, 400000)

