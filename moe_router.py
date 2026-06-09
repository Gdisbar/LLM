import torch
import torch.nn.functional as F

"""
1. Compute Routing Probabilities (Softmax):
Apply the softmax function to the gate_logits along the expert 
dimension (dimension 1). This converts raw scores into a probability 
distribution for each token that sums to 1.

2. Determine Hard Routing (Argmax):
For each token, determine which expert has the highest probability. 
Use torch.argmax on the logits (or the probabilities) along the expert 
dimension. This creates a mask or an index tensor of shape (num_tokens,) 
identifying which expert received each token.

3. Calculate Dispatch Fraction (f):
For every expert, calculate the fraction of total tokens it received.
Calculation: Count how many times each expert index appears in your 
argmax result, then divide by the total number of tokens. This results 
in a vector of length num_experts.

4. Calculate Average Gating Probability (P):
For every expert, calculate the mean of the probabilities assigned to 
it by the gating network across all tokens.
Calculation: Sum the probabilities assigned to expert i across all 
tokens and divide by the total number of tokens. This results in a vector 
of length num_experts.

5. Compute the Final Loss:
Apply the formula provided:Loss = alpha * N * Σ(1,n) f_i * P_i
Where N = num_experts, f_i = dispatch fraction for expert i, and 
P_i = average gating probability for expert i.


"""
def moe_load_balancing_loss(gate_logits, num_experts: int, alpha: float = 0.01) -> float:
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: torch.Tensor of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    if gate_logits.dtype!=torch.tensor:
        gate_logits = torch.tensor(gate_logits,dtype=torch.float32)
    # Compute Routing Probabilities (Softmax)
    routing_probs = F.softmax(gate_logits,dim=-1) # (num_tokens, num_experts)
    # Determine Hard Routing (Argmax)
    assignments = torch.argmax(routing_probs,dim=-1) # (num_tokens,) 
    # Calculate Dispatch Fraction (f)
    # Count occurrences of each expert assignment
    expert_counts = torch.bincount(assignments,minlength=num_experts) # (num_experts) 
    f = expert_counts.float()/gate_logits.size(0) # num_tokens
    # Calculate Average Gating Probability (P)
    # Sum probabilities for each expert across tokens 
    # (num_tokens, num_experts) -> (num_experts)
    expert_probs_sum = torch.sum(routing_probs,dim=0) 
    P = expert_probs_sum/gate_logits.size(0)
    # Compute the Final Loss
    loss = alpha * num_experts * torch.sum(f*P,dim=0)
    return round(loss.item(),4)






result = moe_load_balancing_loss(torch.tensor([[10.0, 0.0], [0.0, 10.0]]), 
                                 num_experts=2, alpha=0.01)
print(result) # 0.01