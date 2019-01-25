import torch


def head_entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def head_pairwise_kl(p):
    # p has shape bsz x nheads x L x L and is normalized in the last
    # dim
    logp = torch.log(p)
    logp[p == 0] = 0
    H_pq = -torch.einsum(
        "blij,bljk->blik",
        [p.permute(0, 2, 1, 3), logp.permute(0, 2, 3, 1)]
    ).permute(0, 2, 3, 1)
    H_p = head_entropy(p).unsqueeze(-2)
    KL = H_pq - H_p
    return KL
