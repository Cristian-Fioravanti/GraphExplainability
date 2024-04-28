from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import numpy as np

def get_attention_scores(model, event):
    model.eval()

    A = to_dense_adj(event.edge_index)[0]
    attention_scores_list = []

    for layer in model.layers:
        h1 = model.apply_embedding(event)
        h, attention_scores = layer.MHGAtt(A, h1)  # Assuming you want to visualize attention scores for Multi-Head Graph Attention
        attention_scores_list.append(attention_scores.detach().numpy())

    num_layers = len(attention_scores_list)
    num_heads = attention_scores_list[0].shape[0]

    # Define node names based on the number of nodes in the graph
    num_nodes = event.num_nodes

    # Calculate the minimum and maximum values of attention scores across all layers and heads
    min_value = min(np.min(scores) for scores in attention_scores_list)
    max_value = max(np.max(scores) for scores in attention_scores_list)

    return attention_scores_list,min_value,max_value, num_nodes, h1, h


def get_graph_for_each_layer(pred):

    # Numero di grafici da creare
    num_graphs = len(pred) // 3

    # Creazione e visualizzazione dei grafici
    for i in range(num_graphs):
        start_idx = i * 3
        end_idx = start_idx + 3
        triplet_pred = pred[start_idx:end_idx]
        
        plt.figure(figsize=(10, 6))
        
        # Traccia i valori
        for idx, y in enumerate(zip(*triplet_pred)):
            x_values = range(len(y))
            plt.plot(x_values, y, label=f'Segnale' if idx == 0 else f'Background')

        # Aggiungi etichette sull'asse x
        plt.xticks(x_values, [f'Layer {i}' for i in range(len(x_values))])
        
        # Aggiungi etichette e titolo
        plt.xlabel('Layer')
        plt.ylabel('Valori')
        plt.title(f'Valori dei layer (Tripletta {i + 1})')
        
        plt.legend()
        plt.grid(True)
        plt.show()

# def get_classification_per_layer(model, event):
#     attention_scores_list,min_value,max_value, num_nodes, h1, h = get_attention_scores(model,event)

#     # Calculate the number of layers
#     num_layers = len(attention_scores_list)

#     # Calculate the number of heads
#     num_heads = attention_scores_list[0].shape[0]

#     prediction_scores_list = []
#     for layer_idx in range(num_layers):
#         # Loop through each head and add its corresponding heatmap to a subplot
#         for head_idx in range(num_heads):
#             scores = attention_scores_list[layer_idx][head_idx]
#             h = model.layer[0].forward_withoutMHFAtt(h1,h)
#             h = model.predict_from_hidden_state(event,h)
#             prediction_scores_list.append(h)
#     return prediction_scores_list