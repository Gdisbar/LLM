import torch
import torch.nn.functional as F
from typing import Tuple
import math
import numpy as np

def compute_qkv(X,W_q, W_k, W_v)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    if X.dtype!=torch.tensor:
        X = torch.tensor(X,dtype=torch.float32)
        W_q = torch.tensor(W_q,dtype=torch.float32)
        W_k = torch.tensor(W_k,dtype=torch.float32)
        W_v = torch.tensor(W_v,dtype=torch.float32)
    q = torch.matmul(X,W_q)
    k = torch.matmul(X,W_k)
    v = torch.matmul(X,W_v)
    return q,k,v

def softmax_logits_probability(x,axis=-1)->torch.Tensor:
	z = x-torch.max(x,dim=axis,keepdim=True).values # shape of x
	e_z = torch.exp(z)    # shape of x
	sm_e_z = torch.sum(e_z,dim=axis,keepdim=True) # shape of x
	return e_z/sm_e_z # shape of x

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention then add mask

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    if Q.dtype!=torch.tensor:
        q = torch.tensor(Q,dtype=torch.float32)
        k = torch.tensor(K,dtype=torch.float32)
        v = torch.tensor(V,dtype=torch.float32)
    else:
         q = torch.clone(Q)
         k = torch.clone(K)
         v = torch.clone(V)
    d_k = k.size(-1) # d_model
    q_k = torch.matmul(q,k.permute(1,0)) # q (seq_len,d_q) X k.T (d_k,seq_len)
    atten_score = q_k/math.sqrt(d_k)
    atten_wt = softmax_logits_probability(atten_score,axis=-1) #  (seq_len,seq_len)
    attn_output = torch.matmul(atten_wt,v) #  (seq_len,d_v)
    return attn_output

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len,d_model = Q.shape
    d_k = d_model // n_heads  # each head dimension
    # (seq_len,n_heads,d_k) -> (n_heads,seq_len,d_k)
    q = Q.reshape(seq_len,n_heads,d_k).permute(1,0,2)
    k = K.reshape(seq_len,n_heads,d_k).permute(1,0,2)
    v = V.reshape(seq_len,n_heads,d_k).permute(1,0,2)
    head_outputs = []
    for i in range(n_heads):
         # q[i],k[i],v[i] -> (seq_len,d_k) 
         head_outputs.append(self_attention(q[i],k[i],v[i]))
    # Concatenate: list of (seq_len, d_k) 
    # stack -> (n_heads, seq_len, d_k)
    # transpose -> (seq_len, n_heads, d_k)
    # reshape -> (seq_len, d_model)
    multi_head_output = torch.stack(head_outputs,dim=0).permute(1,0,2).reshape(seq_len, d_model)
    return multi_head_output


np.random.seed(42)
X = np.random.permutation(np.arange(16)).reshape(4, 4)
W_q = np.random.randint(0, 4, size=(4, 4))
W_k = np.random.randint(0, 5, size=(4, 4))
W_v = np.random.randint(0, 6, size=(4, 4))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
result = multi_head_attention(Q, K, V, n_heads=2)
print(torch.round(result,decimals=0).to(torch.int16).tolist()) 


# [[103, 109, 46, 99], [103, 109, 46, 99], [103, 109, 46, 99], [103, 109, 46, 99]]