import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGraphAttention(nn.Module):
    """Multi-Head Graph Attention Module"""

    def __init__(self, hidden_size=40, num_heads=3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        #define projection
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h): #A:adiacency matrix -- h: features of graphs
        N = h.size(0)  # Number of nodes

        # Compute query keys and value as projection of the input
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        #resize with dimension (num_heads, N, head_size)
        q = q.view(N, self.num_heads, self.head_size).transpose(0, 1)  # (num_heads, N, head_size)
        k = k.view(N, self.num_heads, self.head_size).transpose(0, 1)  # (num_heads, N, head_size)
        v = v.view(N, self.num_heads, self.head_size).transpose(0, 1)  # (num_heads, N, head_size)

     
        #compute attention scores
        scores = torch.matmul(q, k.transpose(1, 2)) * A.unsqueeze(0) / (N ** (-0.5)) #(num_heads, N, N) attention score between a pair of nodes for each attention head
        scores = F.softmax(scores, dim=2)

        out = torch.matmul(scores, v) #(num_heads, N, head_size)
        out = out.transpose(0, 1).contiguous().view(N, self.hidden_size)  # (N, hidden_size)

        return out, scores