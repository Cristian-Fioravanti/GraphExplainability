#Un evento e' un grafo formato da 6 o 7 nodi e ogni nodo ha n feature (con n che arriva a 12 dopo la manipolazione fatta da lei)
import matplotlib.pyplot as plt
import torch
from sparticles.transforms import MakeHomogeneous
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch_geometric.transforms import BaseTransform
import numpy as np
import plotly.graph_objects as go
from GraphModel.GraphTransformerModel import GraphTransformerModel
from data.datasetClass import CustomEventsDataset
from utils import get_attention_scores, get_graph_for_each_layer
from plotly.subplots import make_subplots

print("Cuda is available: " + str(torch.cuda.is_available()))

def printAttentionScoreAsHeatMap(attention_scores_list,min_value,max_value,num_nodes):
    if num_nodes == 7:
        node_names = ['jet1', 'jet2', 'jet3', 'b1', 'b2', 'lepton', 'energy']
    elif num_nodes == 6:
            node_names = ['jet1', 'jet2', 'b1', 'b2', 'lepton', 'energy']
    else:
            raise ValueError("Unsupported number of nodes in the graph")

    # Calculate the number of layers
    num_layers = len(attention_scores_list)

    # Calculate the number of heads
    num_heads = attention_scores_list[0].shape[0]

    # Create subplots with num_layers rows and num_heads columns
    fig = make_subplots(rows=num_layers, cols=num_heads, subplot_titles=[f'Head {h+1}' for l in range(num_layers) for h in range(num_heads)])

    # Loop through each layer
    for layer_idx in range(num_layers):
        # Loop through each head and add its corresponding heatmap to a subplot
        for head_idx in range(num_heads):
            scores = attention_scores_list[layer_idx][head_idx]
            heatmap = go.Heatmap(z=scores[::-1, :], colorscale='viridis', zmin=min_value, zmax=max_value)
            fig.add_trace(heatmap, row=layer_idx + 1, col=head_idx + 1)

            # Update layout for each subplot
            fig.update_xaxes(tickmode='array', tickvals=np.arange(len(node_names)), ticktext=node_names, row=layer_idx + 1, col=head_idx + 1)
            fig.update_yaxes(tickmode='array', tickvals=np.arange(len(node_names)), ticktext=list(reversed(node_names)), row=layer_idx + 1, col=head_idx + 1)

    # Update the layout for the entire figure
    fig.update_layout(height=500 * num_layers, width=1500, title_text='Attention Scores Heatmap', title_x=0.5)

    fig.show()

config=dict(
      out_size = 2,
      num_layers=2,
      hidden_size=60,
      input_size=12,
      num_heads= 4,
      learning_rate = 0.0005,
      weight_decay=0.0005,
      batch_size = 1,
      signal=400000,
      singletop=200000,
      ttbar=200000,
      dropout = 0.3,
      normalization = True
)
checkpoint_path = './checkpoint/'

print("set config: "+ str(config))

print("load dataset...")

dataset = CustomEventsDataset(
    root='./data',
    url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
    delete_raw_archive=False,
    add_edge_index=True,
    event_subsets={'signal': config['signal'], 'singletop': config['singletop'], 'ttbar': config['ttbar']},
    transform=MakeHomogeneous()
)

#split the dataset
print("Preparing training and testing sets")
# generate indices: instead of the actual data we pass in integers
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    train_size=0.8,
    stratify=[g.y.item() for g in dataset], # to have balanced subsets
    random_state=42
)

dataset_train = Subset(dataset, train_indices)
dataset_test = Subset(dataset, test_indices)

print(f'Train set contains {len(dataset_train)} graphs, Test set contains {len(dataset_test)} graphs')

# Dataloaders
train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
model = GraphTransformerModel(out_size=config['out_size'],
                              input_size=config['input_size'],
                              hidden_size=config['hidden_size'],
                              num_layers=config['num_layers'],
                              num_heads=config['num_heads'],
                              dropout=config['dropout'],
                              normalization=config['normalization']).to(device)

# Load the trained weights onto the model
checkpoint = torch.load(checkpoint_path+"checkpoint_epoch_001_2l.pt", map_location=device)

model.load_state_dict(checkpoint)

# Make predictions on the test dataset
predictions = []
targets = []
predictionsCase1 = []
targetsCase1 = []
predictionsCase2 = []
targetsCase2 = []
for index_x, data in enumerate(test_loader):
    originalGraph = data.to(device)
    print("originalGraph: " + str(originalGraph)) #originalGraph: DataBatch(x=[7, 12], edge_index=[2, 42], y=[1], event_id=[1], data_norm=[7, 12], batch=[7], ptr=[2])
    #fase 0 
    h = model.apply_embedding(originalGraph)
    pred = model.predict_from_residual_connection(originalGraph,h)
    predictions.extend(pred.tolist())
    targets.extend(data.y.tolist())

    for index_layer in range(config['num_layers']):
        #fase layer ---> index_layer
        h = model.apply_GTLayer(originalGraph,h,index_layer)
        pred = model.predict_from_residual_connection(originalGraph,h)
        predictions.extend(pred.tolist())
        targets.extend(data.y.tolist())

    # Case 1
    out = model.forward_with_no_grad(originalGraph)
    pred = out.argmax(dim=1)
    predictionsCase1.extend(out.tolist())
    targetsCase1.extend(originalGraph.y.tolist())
    # Case 2
    for index, patch in enumerate(originalGraph.x):
        patchGraph = originalGraph
        patchGraph.x = patch
        out = model.forward_with_no_grad(patchGraph)
        pred = out.argmax(dim=1)
        predictionsCase2.extend(out.tolist())
        targetsCase2.extend(patchGraph.y.tolist())
    

    print("predictions: " + str(predictions))
    print("targets: " + str(targets))
    print("predictions Case1: " + str(predictionsCase1))
    print("targets Case1: " + str(targetsCase1))
    print("predictions Case2: " + str(predictionsCase2))
    print("targets Case2: " + str(targetsCase2))

    get_graph_for_each_layer(predictions)
    break
 
# Cuda is available: True
# set config: {'out_size': 2, 'num_layers': 2, 'hidden_size': 60, 'input_size': 12, 'num_heads': 4, 'learning_rate': 0.0005, 'weight_decay': 0.0005, 'batch_size': 1, 'signal': 400000, 'singletop': 200000, 'ttbar': 200000, 'dropout': 0.3, 'normalization': True}
# load dataset...
# Preparing training and testing sets
# Train set contains 640000 graphs, Test set contains 160000 graphs
# originalGraph: DataBatch(x=[7, 12], edge_index=[2, 42], y=[1], event_id=[1], data_norm=[7, 12], batch=[7], ptr=[2])
# predictions: [[0.31151071190834045, -0.11943435668945312], [0.57164466381073, -0.4263044595718384], [0.7678094506263733, -0.6231948733329773]]
# targets: [0, 0, 0]
# predictions Case1: [[0.3739151656627655, -0.181740865111351]]
# targets Case1: [0]
# predictions Case2: [[0.479033887386322, -0.2858933210372925], [0.590991199016571, -0.4133935570716858], [0.6525576710700989, -0.4843935966491699], [0.43780767917633057, -0.2531468868255615], [0.519417405128479, -0.3680572509765625], [0.8329042196273804, -0.708314836025238], [0.4984736442565918, -0.31823229789733887]]
# targets Case2: [0, 0, 0, 0, 0, 0, 0]

# *********-----------------------------------------------------------------------------*******

# Cuda is available: True
# set config: {'out_size': 2, 'num_layers': 2, 'hidden_size': 60, 'input_size': 12, 'num_heads': 4, 'learning_rate': 0.0005, 'weight_decay': 0.0005, 'batch_size': 1, 'signal': 400000, 'singletop': 200000, 'ttbar': 200000, 'dropout': 0.3, 'normalization': True}
# load dataset...
# Preparing training and testing sets
# Train set contains 640000 graphs, Test set contains 160000 graphs
# originalGraph: DataBatch(x=[7, 12], edge_index=[2, 42], y=[1], event_id=[1], data_norm=[7, 12], batch=[7], ptr=[2])
# predictions: [[0.31151071190834045, -0.11943435668945312], [0.5414880514144897, -0.3871431350708008], [0.45069658756256104, -0.2734595537185669]]
# targets: [0, 0, 0]
# predictions Case1: [[0.6312587261199951, -0.45628464221954346]]
# targets Case1: [0]
# predictions Case2: [[0.23220285773277283, -0.031628236174583435], [0.5241447687149048, -0.3406686782836914], [0.5542008280754089, -0.38228797912597656], [0.6694334149360657, -0.507979154586792], [0.5483032464981079, -0.38751471042633057], [0.6456649899482727, -0.49834108352661133], [0.7315323948860168, -0.5837251543998718]]
# targets Case2: [0, 0, 0, 0, 0, 0, 0]



















# 
    # currentIndex = 0


        
        
#         #Compute model result
#         out = model(patchGraph)
#         pred = out.argmax(dim=1)
#         print("pred model" + str(pred))
#         predictions.extend(pred.tolist())
#         targets.extend(originalGraph.y.tolist())
        
#         #Compute the attention scores of the event
#         attention_scores_list,min_value,max_value, num_nodes = get_attention_scores(model,patchGraph)
#         printAttentionScoreAsHeatMap(attention_scores_list,min_value,max_value,num_nodes)
#     if (index_x == 200):
#         break
# print("predictions: " + str(predictions))
# print("targets: " + str(targets))

# # # Calculate confusion matrix
# conf_matrix = confusion_matrix(targets, predictions)

# # Plot confusion matrix with correct labels
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['background', 'signal'], yticklabels=['background', 'signal'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

