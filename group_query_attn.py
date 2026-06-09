import torch
import torch.nn. functional as F
import numpy as np
import math

"""
In GQA, query heads are divided into groups, and each group of query heads 
shares the same key and value head. This reduces the memory footprint of the 
KV cache during inference while retaining most of the quality benefits of 
full multi-head attention.

Special Cases:
When num_heads == num_kv_heads, GQA reduces to standard Multi-Head Attention.
When num_kv_heads == 1, GQA reduces to Multi-Query Attention.

`num_heads` must be evenly divisible by `num_kv_heads`

"""

def secaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention 

    Args:
        Q: Query matrix of shape (batch,num_heads,seq_len,head_dim_Q)
        K: Key matrix of shape (batch,num_kv_heads,seq_len,head_dim_K)
        V: Value matrix of shape (batch,num_kv_heads,seq_len,head_dim_V)

    Returns:
        Attention output of shape (batch,num_heads,seq_len,head_dim_V)
    """
    d_k = Q.size(-1) # head_dim
    # Q (batch,num_heads,seq_len,head_dim) 
    # K (batch,num_kv_heads,seq_len,head_dim) 
    # K.T -> (batch,num_kv_heads,head_dim,seq_len)
    q_k = torch.matmul(Q,K.permute(0,1,3,2)) # (batch,num_heads,seq_len,seq_len)
    atten_score = q_k/math.sqrt(d_k)
    # on seq_len : (batch,num_heads,seq_len,seq_len)
    atten_wt = F.softmax(atten_score,dim=-1) 
    attn_output = torch.matmul(atten_wt,V) #  (batch,num_heads,seq_len,head_dim_V)
    return attn_output

def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, 
                            V: torch.Tensor, num_heads: int, 
                            num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    
    batch_size, seq_len, d_model = Q.shape
    head_dim = K.shape[2] // num_kv_heads
    # reshape Q,K,V
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K = K.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    V = V.reshape(batch_size, seq_len, num_kv_heads, head_dim)
    # num_groups : how many query heads share one KV head
    num_groups = num_heads // num_kv_heads
    # Expand K and V to match num_heads
    K = K.repeat_interleave(num_groups,dim=2)
    V = V.repeat_interleave(num_groups,dim=2)
    # transpose to (batch,seq_len,num_heads,head_dim) -> (batch,num_heads,seq_len,head_dim)
    Q = Q.permute(0,2,1,3)
    K = K.permute(0,2,1,3)
    V = V.permute(0,2,1,3)
    # scaled-dot-product-attention - (batch,num_heads,seq_len,head_dim_V)
    scdp_output = secaled_dot_product_attention(Q,K,V)
    # (batch,num_heads,seq_len,head_dim_V) 
    # -> (batch,seq_len,num_heads,head_dim_V) -> (batch,seq_len,d_model)
    gqa_output = scdp_output.permute(0,2,1,3).reshape(batch_size,seq_len,d_model)
    return gqa_output



Q = torch.tensor([[[1.0, 0.0, 0.5, 0.5]]])
K = torch.tensor([[[1.0, 1.0]]])
V = torch.tensor([[[2.0, 3.0]]])
result = grouped_query_attention(Q, K, V, num_heads=2, num_kv_heads=1)
print(torch.round(result, decimals=4))

# [[[2. 3. 2. 3.]]]