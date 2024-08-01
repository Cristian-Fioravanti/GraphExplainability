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
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self, h):  # A: adjacency matrix -- h: features of graphs
        batch_size = h.size(0)  # Number of nodes

        # Compute query, keys and values as projection of the input
        q = self.q_proj(h)  # Shape: (N, num_heads * head_size)
        k = self.k_proj(h)  # Shape: (N, num_heads * head_size)
        v = self.v_proj(h)  # Shape: (N, num_heads * head_size)

        # Divisione in num_heads
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)

        # Calcolo dell'attenzione
        scores = self.calculate_attention(q, k, v)

        # Concatenazione delle teste
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)  # (batch_size, seq_length, hidden_size)
        
        # Livello lineare finale
        output = self.out_proj(concat)  # (batch_size, seq_length, hidden_size)

         # Normalizzazione del layer
        output = self.layer_norm(output + h)  # Aggiungi il residuo e normalizza

        return output
       

    def calculate_attention(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output