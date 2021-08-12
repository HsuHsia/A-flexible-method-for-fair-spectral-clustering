import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv('dataset/Facebook-known-pairs_data_2013.csv')
# 变成List
connections = tuple(list(df['1 984 0']))

G = nx.Graph()
print(G)
for i in range(len(connections)):
    connec = int(connections[i].split(" ")[2])
    if connec == 1:
        vertice_string = int(connections[i].split(" ")[0]) # to get the first element of the string
        vertice_connection = int(connections[i].split(" ")[1]) # to get the second element of the string
        G.add_edge(vertice_string, vertice_connection)
        
nx.draw_networkx(G)

laplacian_matrix = nx.laplacian_matrix(G)
print(laplacian_matrix)