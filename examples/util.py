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


def parse_head_pruning_descriptors(
    descriptors,
    reverse_descriptors=False,
    n_heads=None
):
    """Returns a dictionary mapping layers to the set of heads to prune in
    this layer"""
    to_prune = {}
    for descriptor in descriptors:
        layer, heads = descriptor.split(":")
        layer = int(layer) - 1
        heads = set(int(head) - 1 for head in heads.split(","))
        if layer not in to_prune:
            to_prune[layer] = set()
        to_prune[layer].update(heads)
    # Reverse
    if reverse_descriptors:
        if n_heads is None:
            raise ValueError("You need to specify the total number of heads")
        for layer, heads in to_prune.items():
            to_prune[layer] = set([head for head in range(n_heads)
                                   if head not in heads])
    return to_prune
