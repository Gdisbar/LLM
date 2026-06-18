import torch
import math
import torch.nn.functional as F


def self_attention(q,k,v)->torch.Tensor:
    """
    Compute scaled dot-product self-attention then add mask

    Args:
        q: Query matrix of shape (d_q,)
        k: Key matrix of shape (seq_len,d_k)
        v: Value matrix of shape (seq_len,d_v)

    Returns:
        Attention output of shape (d_v,)
    """

    d_k = k.size(-1) # d_model
    q_k = torch.matmul(q,k.permute(1,0)) # q (d_q) X k.T (d_k,seq_len)
    atten_score = q_k/math.sqrt(d_k) #  (seq_len,)
    atten_wt = F.softmax(atten_score,dim=-1) #  (seq_len,)
    attn_output = torch.matmul(atten_wt,v) #  (seq_len,)X(seq_len,d_v)
    return attn_output


def kv_cache_attention_step(x_new: torch.Tensor, 
                            W_Q: torch.Tensor, W_K: torch.Tensor,
                            W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    q = torch.matmul(x_new,W_Q) # (d_model,)
    k_new = torch.matmul(x_new,W_K) # (d_model,)
    v_new = torch.matmul(x_new,W_V) # (d_model,)
    if cache is None:
        K_cache = k_new.unsqueeze(0) # (1,d_model)
        V_cache = v_new.unsqueeze(0) # (1,d_model)
    else:
        K_cache,V_cache = cache
        K_cache = torch.cat([K_cache,k_new.unsqueeze(0)]) # (seq_len,d_model)
        V_cache = torch.cat([V_cache,v_new.unsqueeze(0)]) # (seq_len,d_model)
        
    attn_output = self_attention(q,K_cache,V_cache)
    return attn_output, (K_cache, V_cache)


import numpy as np
W_Q = torch.eye(2)
W_K = torch.eye(2)
W_V = torch.eye(2)
x1 = torch.tensor([1.0, 0.0],dtype=torch.float32)
out1, c1 = kv_cache_attention_step(x1, W_Q, W_K, W_V, None)
print(torch.round(out1,decimals=4).tolist())

# [1.0, 0.0]