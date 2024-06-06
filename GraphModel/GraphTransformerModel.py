from GraphModel.GraphTransformerLayer import GTLayer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import global_mean_pool
import torch
class GraphTransformerModel(nn.Module):
    def __init__(self, out_size, input_size=12, hidden_size=40, num_layers=4, num_heads=3,dropout=0.3, normalization=True):
        super(GraphTransformerModel, self).__init__()

        self.normalization = normalization

        #embedding the input in hidden dim
        self.embedding = nn.Linear(input_size, hidden_size)

        #define the self attention layers
        self.layers = nn.ModuleList([GTLayer(hidden_size, num_heads,dropout) for _ in range(num_layers)])

        # Linear layers for prediction
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.gelu1 = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.gelu2 = nn.GELU()
        self.aggregate = global_mean_pool
        self.linear3 = nn.Linear(hidden_size // 2, out_size)

    def forward(self, data, batch=None):
        A = to_dense_adj(data.edge_index)[0]                # Convert edge index to adjacency matrix
        if self.normalization == True:                      #If normalization is true normalize the data
           h = self.embedding(data.data_norm)
        else:
           h = self.embedding(data.x)

        #Compute GT layers
        for layer in self.layers:
            h = layer(A, h)

        # Linear layers for prediction
        h = self.linear1(h)
        h = self.gelu1(h)
        h = self.linear2(h)
        h = self.gelu2(h)
        # Control where you have to aggregate
        h = self.aggregate(h, data.batch)
        h = self.linear3(h)

        return h

    #method used in the explainability part
    def apply_embedding(self, input_data):
        with torch.no_grad():
            # Apply the learned embedding transformation to input_data.x
            if self.normalization == True:
               embedded_data = self.embedding(input_data.data_norm)
            else:
               embedded_data = self.embedding(input_data.x)
            h = embedded_data
        return h
    
    def apply_GTLayer(self,input_data,h,index_layer):
        with torch.no_grad():
            A = to_dense_adj(input_data.edge_index)[0]
            return self.layers[index_layer](A, h)
        
    def predict_from_residual_connection(self, input_data, h):
        with torch.no_grad():
            # Linear layers for prediction
            h = self.linear1(h)
            h = self.gelu1(h)
            h = self.linear2(h)
            h = self.gelu2(h)
            # Control where you have to aggregate
            h = self.aggregate(h, input_data.batch)
            h = self.linear3(h)
            return h
        
    def forward_with_no_grad(self, data, batch=None):
        A = to_dense_adj(data.edge_index)[0]                # Convert edge index to adjacency matrix
        if self.normalization == True:                      #If normalization is true normalize the data
           h = self.embedding(data.data_norm)
        else:
           h = self.embedding(data.x)

        #Compute GT layers
        for layer in self.layers:
            h = layer(A, h)

        # Linear layers for prediction
        h = self.linear1(h)
        h = self.gelu1(h)
        h = self.linear2(h)
        h = self.gelu2(h)
        # Control where you have to aggregate
        h = self.aggregate(h, data.batch)
        h = self.linear3(h)

        return h