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
from utils import get_attention_scores
from plotly.subplots import make_subplots
config=dict(
      out_size = 2,
      num_layers=2,
      hidden_size=60,
      input_size=12,
      num_heads= 4,
      learning_rate = 0.0005,
      weight_decay=0.0005,
      batch_size = 1,
      dropout = 0.3,
      signal=400000,
      singletop=200000,
      ttbar=200000,
      normalization = True
)
local_path = './checkpoint/checkpoint_epoch_10.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the model
model = GraphTransformerModel(out_size=config['out_size'],
                              input_size=config['input_size'],
                              hidden_size=config['hidden_size'],
                              num_layers=config['num_layers'],
                              num_heads=config['num_heads'],
                              dropout=config['dropout'],
                              normalization=config['normalization']).to(device)

checkpoint_path = './checkpoint/'

# Load the trained weights onto the model
checkpoint = torch.load(checkpoint_path+"checkpoint_epoch_001_2l.pt", map_location=device)

model.load_state_dict(checkpoint)
print(model)