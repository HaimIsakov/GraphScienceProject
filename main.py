import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx


def load_files(file_path):
    col_names = ["from_code", "to_code", "from_airport", "to_airport", "people"]
    graph_df = pd.read_csv(file_path, names=col_names)
    graph_df["people"] = graph_df["people"].astype(int)
    return graph_df


def create_graph(graph_df):
    graph = nx.DiGraph()
    from_airport = list(graph_df["from_code"])
    to_airport = list(graph_df["to_code"])
    people = list(graph_df["people"])
    edges_tuples = [(a, b, c) for a, b, c in zip(from_airport, to_airport, people)]
    graph.add_weighted_edges_from(edges_tuples)
    return graph


def graph_setup(file_path):
    graph_df = load_files(file_path)
    graph = create_graph(graph_df)
    return graph


def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, 10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_path = os.path.join("Data", "links_0.95.csv")
    graph = graph_setup(file_path)
    plot_degree_dist(graph)
    print()
