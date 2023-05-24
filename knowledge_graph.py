import networkx as nx
import matplotlib.pyplot as plt

graph = nx.Graph()
def produce_graph(dict):
    for key in dict:
        graph.add_edge(key, dict[key])
    nx.draw(graph, with_labels=True)
    plt.show()
