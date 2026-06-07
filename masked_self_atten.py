import numpy as np
import torch
import math


def softmax_logits_probability(x,axis=-1)->torch.Tensor:
	z = x-torch.max(x,dim=axis,keepdim=True).values # shape of x
	e_z = torch.exp(z)    # shape of x
	sm_e_z = torch.sum(e_z,dim=axis,keepdim=True) # shape of x
	return e_z/sm_e_z # shape of x

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

def masked_attention(Q, K, V, Mask)->torch.Tensor:
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
        mask = torch.tensor(Mask,dtype=torch.float32)
    else:
         q = torch.clone(Q)
         k = torch.clone(K)
         v = torch.clone(V)
         mask = torch.clone(Mask)
    d_k = k.size(-1) # d_model
    q_k = torch.matmul(q,k.permute(1,0)) # q (seq_len,d_q) X k.T (d_k,seq_len)
    atten_score = q_k/math.sqrt(d_k)
    atten_score_masked = atten_score + mask #  (seq_len,seq_len)
    atten_wt = softmax_logits_probability(atten_score_masked,axis=-1) #  (seq_len,seq_len)
    attn_output = torch.matmul(atten_wt,v) #  (seq_len,d_v)
    return attn_output
	

np.random.seed(42)
X = np.arange(48).reshape(6,8)
X = np.random.permutation(X.flatten()).reshape(6, 8)
mask = np.triu(np.ones((6, 6))*(-np.inf), k=1)
W_q = np.random.randint(0,4,size=(8,8))
W_k = np.random.randint(0,5,size=(8,8))
W_v = np.random.randint(0,6,size=(8,8))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
print(masked_attention(Q, K, V, mask))


# Create a tensor of shape (6, 8) with values 0 to 47
X = torch.arange(48).reshape(6, 8)
# Flatten, shuffle, and reshape back to (6, 8)
# torch.randperm(n) returns a random permutation of integers from 0 to n-1
# We use this as indices to shuffle the flattened tensor
flat_X = X.flatten()
X = flat_X[torch.randperm(flat_X.size(0))].reshape(6, 8)
# Create a (6, 6) tensor filled with -inf and apply upper triangular masking
# Note: k=1 makes the diagonal zero, while everything above it becomes -inf
mask = torch.triu(torch.ones(6, 6) * float('-inf'), diagonal=1)
# Generate a random integer tensor with values in range [0, 4)
# The size argument in PyTorch is a tuple (rows, cols)
W_q = torch.randint(0, 4, size=(8, 8))


# [[547. 490. 399. 495. 485. 439. 645. 393.]
#  [547. 490. 399. 495. 485. 439. 645. 393.]
#  [471. 472. 429. 538. 377. 450. 531. 362.]
#  [471. 472. 429. 538. 377. 450. 531. 362.]
#  [471. 472. 429. 538. 377. 450. 531. 362.]
#  [471. 472. 429. 538. 377. 450. 531. 362.]]