import math
import networkx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_to_random(graph, title):
    clustering_coef = networkx.average_clustering(graph)
    print("Original clustering_coef", clustering_coef)

    average_shortest_path_length = networkx.average_shortest_path_length(graph)
    print("Original average_shortest_path_length", average_shortest_path_length)

    degrees = [val for (node, val) in graph.degree()]
    degrees_df = pd.DataFrame(degrees, columns=['Degree'])
    degrees_df["Degree"].hist(grid=False, bins=15, edgecolor='black')
    plt.title(f"Original {title}")
    plt.xlabel("Degree")
    plt.ylabel("#nodes")
    plt.savefig(f"Original_{title}.png")
    plt.show()

    n = graph.number_of_nodes()
    density = networkx.density(graph)
    g = networkx.gnp_random_graph(n, density)
    clustering_coef = networkx.average_clustering(g)
    print("Random clustering_coef", clustering_coef)
    average_shortest_path_length = networkx.average_shortest_path_length(g)
    print("Random average_shortest_path_length", average_shortest_path_length)

    degrees = [val for (node, val) in g.degree()]
    degrees_df = pd.DataFrame(degrees, columns=['Degree'])
    degrees_df["Degree"].hist(grid=False, bins=15, edgecolor='black')
    plt.title(f"Random {title}")
    plt.xlabel("Degree")
    plt.ylabel("#nodes")
    plt.savefig(f"Random_{title}.png")
    plt.show()
