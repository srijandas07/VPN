#code for computing adjacency matrix for NTU/ Kinect-V2
import numpy as np


def compute_adjacency(dataset_name, alpha, beta):
    if dataset_name=='NTU':
        #adj = np.ones([25, 25])
        adj = np.zeros([25, 25])
        intrinsic_connections = [[1,2], [1,17], [1,13], [2,21], [17,18], [13,14], [18,19], [19,20], [14,15], [15,16],
                                 [21,5], [21,9], [21,3], [3,4], [9,10], [5,6], [6,7], [10,11], [11,12], [12,25],
                                 [12,24], [7,8], [8,23], [8,22]]
        extrinsic_connections = [[12,19], [10,18], [6,14], [8,15], [12,8], [10,6]]
        for connection in intrinsic_connections:
            adj[connection[0] - 1][connection[1] - 1] = alpha
            adj[connection[1] - 1][connection[0] - 1] = alpha

        for connection in extrinsic_connections:
            adj[connection[0] - 1][connection[1] - 1] = beta
            adj[connection[1] - 1][connection[0] - 1] = beta

        for connection in range(0,25):
            adj[connection][connection] = 0
    else:
        adj = np.zeros([25,25])
    return adj


def compute_adjacency_two_person(dataset_name, alpha, beta):
    if dataset_name=='NTU':
        #adj = np.ones([25, 25])
        adj = np.zeros([50, 50])
        intrinsic_connections = [[1,2], [1,17], [1,13], [2,21], [17,18], [13,14], [18,19], [19,20], [14,15], [15,16],
                                 [21,5], [21,9], [21,3], [3,4], [9,10], [5,6], [6,7], [10,11], [11,12], [12,25],
                                 [12,24], [7,8], [8,23], [8,22]]
        extrinsic_connections = [[12,19], [10,18], [6,14], [8,15], [12,8], [10,6]]
        for connection in intrinsic_connections:
            adj[connection[0] - 1][connection[1] - 1] = alpha
            adj[connection[1] - 1][connection[0] - 1] = alpha
            adj[connection[0] - 1 + 25][connection[1] - 1 + 25] = alpha
            adj[connection[1] - 1 + 25][connection[0] - 1 + 25] = alpha

        for connection in extrinsic_connections:
            adj[connection[0] - 1][connection[1] - 1] = beta
            adj[connection[1] - 1][connection[0] - 1] = beta
            adj[connection[0] - 1 + 25][connection[1] - 1 + 25] = beta
            adj[connection[1] - 1 + 25][connection[0] - 1 + 25] = beta

        for connection in range(0,50):
            adj[connection][connection] = 0
    else:
        adj = np.zeros([50,50])
    return adj

def compute_adjacency_directed(dataset_name, alpha, beta):
    if dataset_name=='NTU':
        #adj = np.ones([25, 25])
        adj = np.zeros([25, 25])
        intrinsic_connections = [[1,2], [1,17], [1,13], [2,21], [17,18], [13,14], [18,19], [19,20], [14,15], [15,16],
                                 [21,5], [21,9], [21,3], [3,4], [9,10], [5,6], [6,7], [10,11], [11,12], [12,25],
                                 [12,24], [7,8], [8,23], [8,22]]
        extrinsic_connections = [[12,19], [10,18], [6,14], [8,15], [12,8], [10,6]]
        for connection in intrinsic_connections:
            adj[connection[0] - 1][connection[1] - 1] = alpha

        for connection in extrinsic_connections:
            adj[connection[0] - 1][connection[1] - 1] = beta

        for connection in range(0,25):
            adj[connection][connection] = 0
    else:
        adj = np.zeros([25,25])
    return adj


def compute_adjacency_ST(dataset_name, alpha, beta, gamma, timesteps):
    if dataset_name=='NTU':
        #adj = np.ones([25, 25])
        adj = np.zeros([750, 750])
        intrinsic_connections = [[1,2], [1,17], [1,13], [2,21], [17,18], [13,14], [18,19], [19,20], [14,15], [15,16],
                                 [21,5], [21,9], [21,3], [3,4], [9,10], [5,6], [6,7], [10,11], [11,12], [12,25],
                                 [12,24], [7,8], [8,23], [8,22]]
        extrinsic_connections = [[12,19], [10,18], [6,14], [8,15], [12,8], [10,6]]
        for connection in intrinsic_connections:
            for t in range(0, timesteps):
                adj[connection[0] - 1 + (25*t)][connection[1] - 1 + (25*t)] = alpha
                adj[connection[1] - 1 + (25*t)][connection[0] - 1 + (25*t)] = alpha

        for connection in extrinsic_connections:
            for t in range(0, timesteps):
                adj[connection[0] - 1 + (25*t)][connection[1] - 1 + (25*t)] = beta
                adj[connection[1] - 1 + (25*t)][connection[0] - 1 + (25*t)] = beta

        for t in range(0, timesteps-1):
            for joints in range(0, 25):
                adj[joints + (25*t)][joints + (25*(t+1))] = gamma
                adj[joints + (25*(t+1))][joints + (25*t)] = gamma

        for connection in range(0,25):
            adj[connection][connection] = 0
    else:
        adj = np.zeros([750,750])
    return adj