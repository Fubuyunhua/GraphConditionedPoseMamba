"""Parameter-neutral selection of graph-conditioned projection slices."""
import torch

def route_projection(content, context, rank, state, target):
    if content.shape != context.shape or content.shape[-2] != rank+2*state:
        raise ValueError('Selective projection shape mismatch')
    if target == 'all':
        return context
    if target == 'none':
        return content
    if target == 'delta':
        return torch.cat((context[..., :rank, :], content[..., rank:, :]),dim=-2)
    if target == 'bc':
        return torch.cat((content[..., :rank, :], context[..., rank:, :]),dim=-2)
    raise ValueError('Unknown graph conditioning target')
