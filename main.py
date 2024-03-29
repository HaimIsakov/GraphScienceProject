import os

from airport_network import *
from compare_to_random import compare_to_random
from random_network import create_random_network
from states_network import *

# Airports Network
file_path = os.path.join("Data", "links_0.95.csv")
nodes_file_path = os.path.join("Data", "nodes_0.95.csv")
airport_graph = AirportGraph(nodes_file_path, file_path)
# compare_to_random(airport_graph.graph, "Airports graph")
# create_random_network(airport_graph.graph)

# airport_graph.conn_between_binary_in_degree_binary_out_degree()
# airport_graph.conn_between_betweenes_weighted_degree()
# airport_graph.conn_between_binary_degree_weighted_degree()
# airport_graph.plot_degree_dist()
# airport_graph.plot_closeness_centrality()
# airport_graph.plot_distance_dist()
# nx.nodes_with_selfloops(airport_graph.graph)
# # graph.nodes_with_selfloops()
# part = community.best_partition(airport_graph.graph, weight='weight')
# k = 10
# # k_top_airports = airport_graph.find_k_hubs(k)
# airport_graph.plot_betweenness_centrality()
# # print(k_top_airports)
# airport_graph.plot_whole_distance_dist()
#
# States Network
state_graph = StatesGraph(nodes_file_path, airport_graph)
x=1
# state_graph.create_states_file(state_graph.states_graph, name='states_graph')
# state_graph.create_states_file(state_graph.normalized_states_graph, name='norm_states_graph')
# compare_to_random(state_graph.states_graph, "States graph")

# create_random_network(state_graph.states_graph)


# print("Most Crowded State - not normalized")
# print(state_graph.find_most_crowded_states_airport(k=10))
#
# print("Most Crowded State - normalized")
# print(state_graph.find_most_crowded_states_airport_normalized(k=10))
# state_graph.calc_correlation_between_normalized_and_non_normalized_passangers()