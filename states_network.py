import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.stats import pearsonr


def load_nodes_file(file_path):
    col_names = ["airport_code", "airport_short_name", "airport_name", "loc1", "loc2", "unknown1", "unknown2", "state", "unknown3", "unknown4", "unknown5", "continent"]
    nodes_graph_df = pd.read_csv(file_path, names=col_names, sep=';', index_col=False)
    nodes_graph_df["airport_code"] = nodes_graph_df["airport_code"].astype(int)
    return nodes_graph_df


class StatesGraph:
    def __init__(self, nodes_file_path, airport_network):
        self.nodes_graph_df = load_nodes_file(nodes_file_path)
        self.id_states_airports_dict = self.create_dict_between_states_and_airports()
        self.states_graph = self.create_state_graph(airport_network)
        self.normalized_states_graph = self.create_normalized_state_graph(airport_network)

    def create_dict_between_states_and_airports(self):
        id_states_airports_dict = {}
        def add_to_dict(id_states_airports_dict, x):
            if x["airport_code"] not in id_states_airports_dict:
                id_states_airports_dict[x["airport_code"]] = x["state"]

        self.nodes_graph_df.apply(lambda x: add_to_dict(id_states_airports_dict, x), axis=1)
        return id_states_airports_dict

    def create_state_graph(self, airport_network):
        print("create state network")
        states_graph = nx.Graph()
        airport_graph = airport_network.graph
        for code_airport_from, code_airport_to, weight_dict in airport_graph.edges(data=True):
            weight = weight_dict['weight']
            state_from = self.id_states_airports_dict[code_airport_from]
            state_to = self.id_states_airports_dict[code_airport_to]
            if state_from not in states_graph.nodes():
                states_graph.add_node(state_from, weight=0)
            if state_to not in states_graph.nodes():
                states_graph.add_node(state_to, weight=0)
            states_graph.nodes[state_from]['weight'] += weight
            states_graph.nodes[state_to]['weight'] += weight
            # if states_graph.nodes[state_to]['weight']
            states_graph.add_edge(state_from, state_to)
        return states_graph

    def create_normalized_state_graph(self, airport_network):
        print("create normalized state network")
        normalized_states_graph = nx.Graph()
        airport_graph = airport_network.graph
        for code_airport_from, code_airport_to, weight_dict in airport_graph.edges(data=True):
            weight = weight_dict['weight']
            state_from = self.id_states_airports_dict[code_airport_from]
            state_to = self.id_states_airports_dict[code_airport_to]
            if state_from not in normalized_states_graph.nodes():
                normalized_states_graph.add_node(state_from, weight=0)
            if state_to not in normalized_states_graph.nodes():
                normalized_states_graph.add_node(state_to, weight=0)
            normalized_states_graph.nodes[state_from]['weight'] += weight
            normalized_states_graph.nodes[state_to]['weight'] += weight
            normalized_states_graph.add_edge(state_from, state_to)
        grouped_by_states = (self.nodes_graph_df.groupby(['state']).count())['airport_code']
        for state, weight_dict in normalized_states_graph.nodes(data=True):
            weight_dict['weight'] = weight_dict['weight'] / grouped_by_states[state]
        return normalized_states_graph

    def find_most_crowded_states_airport_normalized(self, k=10):
        return sorted(list(self.normalized_states_graph.nodes(data=True)), key=lambda x: x[1]['weight'], reverse=True)[:k]

    def find_most_crowded_states_airport(self, k=10):
        return sorted(list(self.states_graph.nodes(data=True)), key=lambda x: x[1]['weight'], reverse=True)[:k]

    def calc_correlation_between_normalized_and_non_normalized_passangers(self):
        a = sorted(list(self.normalized_states_graph.nodes(data=True)), key=lambda x: x[0], reverse=True)
        b = sorted(list(self.states_graph.nodes(data=True)), key=lambda x: x[0], reverse=True)
        a_values = [np.log(x[1]['weight']) for x in a]
        b_values = [np.log(x[1]['weight']) for x in b]
        pearson_corr, pvalue = pearsonr(a_values, b_values)
        print(f"Pearson Correlation is {pearson_corr}")

    def create_dict_between_states_and_continent(self):
        states_and_continent = {}

        def add_to_dict(states_and_continent, x):
            if x["state"] not in states_and_continent:
                states_and_continent[x["state"]] = x["continent"]

        self.nodes_graph_df.apply(lambda x: add_to_dict(states_and_continent, x), axis=1)
        return states_and_continent

    def create_states_file(self, graph, name='states_graph'):
        nodes = graph.nodes(data=True)
        nodes_idx = dict([(node[0], i) for i, node in enumerate(nodes)])
        continent_dict = self.create_dict_between_states_and_continent()
        nodes = [[nodes_idx[state], state, data['weight'], continent_dict[state]] for state, data in nodes]
        edges = list(graph.edges)
        edges = [[nodes_idx[node1], nodes_idx[node2], node1, node2] for node1, node2 in edges]
        self.save_to_file([['id', 'state', 'weight', 'continent']] + nodes, name+"_nodes.csv")
        self.save_to_file([['Source', 'Target', 'from', 'to']] + edges, name+"_edges.csv")


    def save_to_file(self, lines_list, path):
        with open(path, 'w', newline='') as file:
            wr = csv.writer(file)
            for line in lines_list:
                wr.writerow(line)
