import matplotlib.pyplot as plt
import networkx as nx

def dibujar_red_neuronal(capas):
    G = nx.DiGraph()
    pos = {}
    colores = []
    
    nombres_capas = ['Entrada\n(4 Features)', 'Capa Oculta\n(10 Neuronas)', 'Salida\n(3 Clases)']
    colores_capa = ['#4A90E2', '#50E3C2', '#F5A623']
    
    node_id = 0
    
    # Crear nodos por capa
    for i, num_nodos in enumerate(capas):
        y_offset = (max(capas) - num_nodos) / 2.0
        for j in range(num_nodos):
            pos[node_id] = (i, j + y_offset)
            G.add_node(node_id, capa=i)
            colores.append(colores_capa[i])
            node_id += 1
            
    # Crear conexiones entre capas consecutivas
    offset_capa = 0
    for i in range(len(capas) - 1):
        nodos_capa_actual = range(offset_capa, offset_capa + capas[i])
        nodos_siguiente_capa = range(offset_capa + capas[i], offset_capa + capas[i] + capas[i+1])
        
        for n1 in nodos_capa_actual:
            for n2 in nodos_siguiente_capa:
                G.add_edge(n1, n2)
                
        offset_capa += capas[i]

    # Dibujar el grafo
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color=colores, node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray')
    
    # Etiquetas de las capas
    for i, nombre in enumerate(nombres_capas):
        plt.text(i, max(capas) + 0.3, nombre, horizontalalignment='center', fontsize=11, fontweight='bold')

    plt.title("Arquitectura Perceptrón Multicapa (MLP) - Dataset Iris", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Estructura de nuestra red: 4 entradas, 10 en capa oculta, 3 salidas
dibujar_red_neuronal([4, 10, 3])