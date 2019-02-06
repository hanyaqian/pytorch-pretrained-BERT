from math import sqrt


def copy_same_uniform(W):
    """Return a copy of the tensor intialized at random with the same
    empirical 1st and second moment"""
    out = W.new_empty(W.size())
    a = (sqrt(3) * W.std()).data
    out.uniform_(-a, a) + W.mean()
    return out


def interpolate_linear_layer(layer, mask, dim=-1, other_layer=None):
    """Interpolates between linear layers.
    If the second layer is not provided, interpolate with random"""
    if other_layer is None:
        W = copy_same_uniform(layer.weight)
        if layer.bias is not None:
            b = copy_same_uniform(layer.bias)
    else:
        W = other_layer.weight
        if layer.bias is not None:
            b = other_layer.bias
    layer.weight.requires_grad = False
    layer.weight.masked_scatter_(mask.unsqueeze(dim), W)
    layer.weight.requires_grad = True
    if layer.bias is not None and dim != 0:
        layer.bias.requires_grad = False
        layer.bias.masked_scatter_(mask, b)
        layer.bias.requires_grad = True
