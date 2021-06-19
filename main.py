import os
import community
from airport_network import *
from states_network import *

# Airports Network
file_path = os.path.join("Data", "links_0.95.csv")
nodes_file_path = os.path.join("Data", "nodes_0.95.csv")
airport_graph = AirportGraph(nodes_file_path, file_path)
# airport_graph.plot_degree_dist()
airport_graph.plot_distance_dist()
# nx.nodes_with_selfloops(airport_graph.graph)
# # graph.nodes_with_selfloops()
# part = community.best_partition(airport_graph.graph, weight='weight')
# k = 10
# # k_top_airports = airport_graph.find_k_hubs(k)
# # airport_graph.plot_betweenness_centrality()
# # print(k_top_airports)

# States Network
state_graph = StatesGraph(nodes_file_path, airport_graph)
print("Most Crowded State - not normalized")
print(state_graph.find_most_crowded_states_airport(k=10))

print("Most Crowded State - normalized")
print(state_graph.find_most_crowded_states_airport_normalized(k=10))