import torch
import torch.nn.functional as F

def jo_criterion(outputs, soft_targets, alpha=1.2, beta=0.8):
    _, C = outputs.size()
    
    p = torch.ones(C, device=outputs.device) / C

    probs = F.softmax(outputs, dim=1)
    avg_probs = torch.mean(probs, dim=0)

    L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
    L_p = -torch.sum(torch.log(avg_probs) * p)
    L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

    loss = L_c + alpha * L_p + beta * L_e
    return probs, loss