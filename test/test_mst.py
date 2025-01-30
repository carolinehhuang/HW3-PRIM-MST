import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    def check_connected(adj_matrix, start):
        nodes = len(adj_matrix)
        visited = []

        stack = [start]
        #implement dfs to check all reachable nodes from any nodes. if the graph is fully connected, the length of the visited nodes will be the same as the total amount of nodes
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.append(current) #append the current node onto the stack
                for neighbor in range(nodes): #traverse through the neighbors
                    if adj_matrix[current][neighbor] != 0 and neighbor not in visited:
                        stack.append(neighbor)

        return len(visited)

    dfs = check_connected(adj_mat, 0)
    assert dfs == len(adj_mat), "Graph is not connected" #check if the number of nodes from dfs is equal to the total amount of nodes

    assert len(mst) == len(adj_mat)  # check if the number of nodes in mst matches number of nodes in adjacency matrix
    assert len(adj_mat) == (np.count_nonzero(mst) / 2) + 1  # check if number of nodes-1 equals to number of connections or in other words, number of nodes is number of edges in mst + 1




def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.

    """
    adj_matrix_unconnected = [
    [0, 5, 0, 0],  # Node 0 is connected to Node 1
    [5, 0, 0, 0],  # Node 1 is connected to Node 0
    [0, 0, 0, 10],  # Node 2 is connected to Node 3
    [0, 0, 10, 0]   # Node 3 is connected to Node 2
]
    g = Graph(np.array(adj_matrix_unconnected))
    g.construct_mst()

    with pytest.raises(AssertionError, match = 'Proposed MST has incorrect expected weight'):
        check_mst(g.adj_mat, g.mst, 15)

    #function would also error at the check_connected step, but fails first at the expected weight due to being unconnected





