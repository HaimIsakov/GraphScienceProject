import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def load_file(file_path):
    col_names = ["from_code", "to_code", "from_airport", "to_airport", "people"]
    graph_df = pd.read_csv(file_path, names=col_names)
    graph_df["people"] = graph_df["people"].astype(int)
    return graph_df


def create_graph(graph_df):
    print("create airport network")
    graph = nx.DiGraph()
    # graph = nx.Graph()
    from_airport = list(graph_df["from_code"])
    to_airport = list(graph_df["to_code"])
    people = list(graph_df["people"])
    # remove self loops
    edges_tuples = [(a, b, c) for a, b, c in zip(from_airport, to_airport, people) if a != b]
    graph.add_weighted_edges_from(edges_tuples)
    return graph


class AirportGraph:
    def __init__(self, edges_file_path):
        self.graph_df = load_file(edges_file_path)
        self.graph = create_graph(self.graph_df)
        self.id_airports_dict = self.create_dict_between_nodes_id_and_airports()

    def create_dict_between_nodes_id_and_airports(self):
        id_airports_dict = {}
        def add_to_dict(id_airports_dict, x):
            if x["from_code"] not in id_airports_dict:
                id_airports_dict[x["from_code"]] = x["from_airport"]
            if x["to_code"] not in id_airports_dict:
                id_airports_dict[x["to_code"]] = x["to_airport"]

        self.graph_df.apply(lambda x: add_to_dict(id_airports_dict, x), axis=1)
        return id_airports_dict

    def find_k_hubs(self, k):
        degrees = np.array([self.graph.degree(n, weight='weight') for n in self.graph.nodes()])
        k_top_indexes = sorted(range(len(degrees)), key=lambda i: degrees[i])[-k:]
        k_top_nodes = [list(self.graph.nodes)[index] for index in k_top_indexes]
        k_top_airports = [self.id_airports_dict[node] for node in k_top_nodes]
        return k_top_airports

    def plot_degree_dist(self):
        log_scale_or_not = True
        degrees = np.array([self.graph.degree(n, weight='weight') for n in self.graph.nodes()])
        if log_scale_or_not:
            # sns.histplot(degrees, bins=200, log_scale=(True, True))
            plt.loglog(degrees, 'bo')
            # plt.yscale('log')
            # plt.xscale('log')
            plt.title('Degrees Histogram (log log scale)')
            plt.tight_layout()
            plt.savefig('Degrees_Histogram_(log log scale)1.png')
            plt.show()
        else:
            sns.histplot(degrees, bins=200, log_scale=(False, False))
            plt.title('Degrees Histogram')
            plt.tight_layout()
            plt.savefig('Degrees_Histogram.png')
            plt.show()

    def plot_betweenness_centrality(self):
        log_scale_or_not = False
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        betweenness_centrality_list = [value for key, value in betweenness_centrality.items() if value >= 1e-300]
        plt.clf()
        if log_scale_or_not:
            sns.histplot(betweenness_centrality_list, bins=200, log_scale=(True, True))
            plt.title('Betweenness Centrality Histogram (log log scale)')
            plt.tight_layout()
            plt.savefig('Betweenness_Centrality_Histogram_(log log scale).png')
        else:
            sns.histplot(betweenness_centrality_list, bins=200, log_scale=(False, False))
            plt.title('Betweenness Centrality Histogram')
            plt.tight_layout()
            plt.savefig('Betweenness_Centrality_Histogram.png')
        plt.show()
