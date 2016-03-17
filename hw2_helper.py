import networkx as nx
import numpy as np


# implement B&A based model with lack of prefferential attachment
def random_graph_model_a(n, m):
    # start with an empty graph of m nodes
    g = nx.empty_graph(m)
    # add other n-m nodes
    for new_node in np.arange(m+1, n+1):
        # add edges from a new node to randomly selected m existing ones
        g.add_edges_from(zip([new_node]*m, random_unique_subset(g.nodes(), m)))
    return g


# same model A but returns plot data to 1.3 task
def random_graph_model_a_average_node_degree_on_iter(n, m):
    # start with an empty graph of m nodes
    g = nx.empty_graph(m)
    # add other n-m nodes
    iters = np.arange(m+1, n+1)
    average_node_degrees = []
    for new_node in np.arange(m+1, n+1):
        # add edges from a new node to randomly selected m existing ones
        g.add_edges_from(zip([new_node]*m, random_unique_subset(g.nodes(), m)))
        average_node_degrees.append(np.average(get_node_degrees(g)))
    return [iters, average_node_degrees]


# implement B&A based model with fixed number of nodes
def random_graph_model_b(n, m):
    # start with empty graph
    g = nx.empty_graph(n)
    for node in g.nodes():
        # chose m nodes from array of nodes each repeated by it's degree+1
        new_edge_targets = random_unique_subset(np.repeat(g.nodes(), np.array(list(g.degree().values()))+1), m)
        # add edges to selected nodes
        g.add_edges_from(zip([node]*m, new_edge_targets))
    return g


def random_graph_model_b_average_node_degree_on_iter(n, m):
    # start with empty graph
    g = nx.empty_graph(n)
    average_node_degrees = []
    for node in g.nodes():
        # chose m nodes from array of nodes each repeated by it's degree+1
        new_edge_targets = random_unique_subset(np.repeat(g.nodes(), np.array(list(g.degree().values()))+1), m)
        # add edges to selected nodes
        g.add_edges_from(zip([node]*m, new_edge_targets))
        average_node_degrees.append(np.average(get_node_degrees(g)))
    return [g.nodes(), average_node_degrees]


# implement vertex copying model
def random_graph_vertex_copying_model(n, q, initial_n=20, initial_p=0.7, is_directed=False):
    # start with simple G_n,p random graph
    g = nx.erdos_renyi_graph(initial_n, initial_p, directed=is_directed)
    for new_node in np.arange(initial_n+1, n+1):
        node_for_copy = np.random.choice(g.nodes())
        neighbors = g.neighbors(node_for_copy)
        non_neighbors = [node for node in g.nodes() if node not in neighbors]
        for neighbor in neighbors:
            # copy edge with probability q
            if np.random.random() <= q:
                g.add_edge(new_node, neighbor)
            # rewire edge to any other node
            else:
                g.add_edge(new_node, np.random.choice(non_neighbors))
    return g


def random_graph_vertex_copying_model_average_node_degree_on_iter(n, q, initial_n=20, initial_p=0.7, is_directed=False,
                                                                  direction=None):
    # start with simple G_n,p random graph
    g = nx.erdos_renyi_graph(initial_n, initial_p, directed=is_directed)
    average_node_degrees = []
    iters = np.arange(initial_n+1, n+1)
    for new_node in np.arange(initial_n+1, n+1):
        node_for_copy = np.random.choice(g.nodes())
        neighbors = g.neighbors(node_for_copy)
        non_neighbors = [node for node in g.nodes() if node not in neighbors]
        for neighbor in neighbors:
            # copy edge with probability q
            if np.random.random() <= q:
                g.add_edge(new_node, neighbor)
            # rewire edge to any other node
            else:
                g.add_edge(new_node, np.random.choice(non_neighbors))
        average_node_degrees.append(np.average(get_node_degrees(g, direction)))
    return [iters, average_node_degrees]


def random_unique_subset(original_set, subset_length):
    if len(original_set) < subset_length:
        return np.unique(original_set)
    subset = set()
    while len(subset) < subset_length:
        elem = np.random.choice(original_set)
        subset.add(elem)
    return subset


def get_node_degrees(g, direction=None):
    if direction == "in":
        return np.array(list(g.in_degree().values()), dtype=np.int)
    elif direction == "out":
        return np.array(list(g.out_degree().values()), dtype=np.int)
    else:
        return np.array(list(g.degree().values()), dtype=np.int)


def get_pdf(node_degrees):
    node_bincount_as_float = np.bincount(node_degrees).astype(float)
    return node_bincount_as_float/len(node_degrees)


def get_cdf(pdf):
    return np.cumsum(pdf)
