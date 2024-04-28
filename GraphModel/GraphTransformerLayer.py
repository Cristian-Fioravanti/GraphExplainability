from GraphModel.MultiHeadGraphAttention import MultiHeadGraphAttention
import torch.nn as nn
import torch.nn.functional as F
import torch

class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=12, num_heads=3, dropout=0.3):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.MHGAtt = MultiHeadGraphAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = dropout
        self.gelu = nn.GELU()

    def forward(self, A, h):
        h1 = h
        h, _ = self.MHGAtt(A, h)  # Compute multi-head graph attention
        h = self.layernorm1(h + h1)  # Add node feature and compute layer norm
        h = F.dropout(h, self.dropout) #Compute dropout

        # Compute feed forward
        h2 = h
        h = self.FFN1(h)
        h = self.gelu(h)
        h = F.dropout(h, self.dropout)
        h = self.FFN2(h)
        h = h2 + h  # Residual connection

        return self.layernorm2(h)  # Layer norm
    
    # def forward_withoutMHFAtt(self,h1,h):
    #     with torch.no_grad():
    #         h = self.layernorm1(h + h1)  # Add node feature and compute layer norm
    #         h = F.dropout(h, self.dropout) #Compute dropout

    #         # Compute feed forward
    #         h2 = h
    #         h = self.FFN1(h)
    #         h = self.gelu(h)
    #         h = F.dropout(h, self.dropout)
    #         h = self.FFN2(h)
    #         h = h2 + h  # Residual connection

    #         return self.layernorm2(h)  # Layer norm