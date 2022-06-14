import pickle
import networkx as nx


def create_random_network(real_graph):
    degrees = [val for (node, val) in real_graph.degree()]
    print("Real degrees", degrees)
    # pickle.dump(real_graph, open("real_graph.pkl", "wb"))
    G = nx.expected_degree_graph(degrees)
    # G = nx.random_degree_sequence_graph(degrees, seed=42, tries=1)
    # pickle.dump(G, open("fake_graph.pkl", "wb"))

    fake_degrees = [val for (node, val) in G.degree()]
    print("Random graph's degrees", fake_degrees)
    print(set(degrees).symmetric_difference(set(fake_degrees)))
    return G
