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

    def forward(self, A, h):  # A: adjacency matrix -- h: features of graphs
        N = h.size(0)  # Number of nodes

        # Compute query, keys and values as projection of the input
        q = self.q_proj(h)  # Shape: (N, num_heads * head_size)
        k = self.k_proj(h)  # Shape: (N, num_heads * head_size)
        v = self.v_proj(h)  # Shape: (N, num_heads * head_size)

        # Resize with dimension (N, num_heads, head_size)
        q = q.view(N, self.num_heads, self.head_size)  # (N, num_heads, head_size)
        k = k.view(N, self.num_heads, self.head_size)  # (N, num_heads, head_size)
        v = v.view(N, self.num_heads, self.head_size)  # (N, num_heads, head_size)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.head_size ** 0.5)  # (N, num_heads, N)
        
        # Softmax on dim=0
        scores = F.softmax(scores, dim=0)  # Apply softmax on the first dimension

        out = torch.matmul(scores, v)  # (N, num_heads, head_size)
        out = out.view(N, self.hidden_size)  # (N, hidden_size)

        return out, scores