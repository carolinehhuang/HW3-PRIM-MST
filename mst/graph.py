import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimu m spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = np.zeros(self.adj_mat.shape) #initialize mst as empty set the same shape as adjacency matrix

        adjacency_mat = self.adj_mat
        num_nodes = len(adjacency_mat)

        visited = [] #initialize visited as empty set
        pq = [(0,0, -1)] #(edge weight, next node, parent node) start at node 0

        while pq:
            weight, node, parent = heapq.heappop(pq) #pop the node with the lowest edge weight from any point in the visited set off of the priority queue
            if node not in visited: #check if node has already been visited
                visited.append(node) #if not visited already, append node to visited set
                if parent != -1: # check if there is a parent (if the node is not the first node)
                    self.mst[node][parent] = weight #if the node is not the first node, set the edge weight between the two points in the matrix
                    self.mst[parent][node] = weight #undirected graph property where mst matrix should be symmetrical

                for neighbor in range(num_nodes): #iterate through all the possible neighbors of the current node
                    edge_weight = adjacency_mat[node][neighbor] #get the edge weight between each possible neighbor
                    if edge_weight != 0 and neighbor not in visited: #check if there is an edge between the neighbors and that the neighbors have not been visited
                        heapq.heappush(pq, (edge_weight, neighbor, node)) #put the neighbor onto the priority queue according to the edge weight






