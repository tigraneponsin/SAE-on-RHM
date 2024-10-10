import torch
import torch.nn as nn
import torch.nn.functional as F


def whiten(tensor, eps):	# subtract meand and divide by std along the batch dimension
    """
    Remove the tensor mean and scale by std along the batch dimension.
    
    Returns:
        Whitened tensor.
    """
    wtensor = torch.clone(tensor)
    return (wtensor-wtensor.mean(dim=0,keepdim=True))/(eps+wtensor.std(dim=0,keepdim=True))


def test( model, dataloader, device):
    """
    Test the model on data from dataloader.
    
    Returns:
        Cross-entropy loss, Classification accuracy.
    """
    model.eval()

    correct = 0
    total = 0
    loss = 0.
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predictions = outputs.max(1)

            loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    return loss / total, 1.0 * correct / total


def sensitivity( model, data, transformed, device):
    """
    Compute the cosine similarity of the model activations when acting on original and transformed data.

    Args:
        model: The model (multilayer attention only).
        data: The original input data.
        transformed: A dictionary with several data transformations.

    Returns:
        Dictionary with model.num_layers+1 entries (one per hidden layer plus one for the output).
    """
    model.eval()

    eps = 1e-8
    result = {}
    B,T,C = data.size()
    # TODO: add batching for when B is too large

    with torch.no_grad():

        act_o = torch.clone(data)	# original activations
        act_o = act_o.to(device)
        act_t = {}                  # transformed activations
        for k in transformed.keys():
            act_t[k] = torch.clone(transformed[k])
            act_t[k] = act_t[k].to(device)

        if hasattr(model, 'token_embedding'):
            act_o = F.linear( act_o, model.token_embedding, bias=None) *C**-.5
            for k in transformed.keys():
                act_t[k] = F.linear( act_t[k], model.token_embedding, bias=None) *C**-.5
            if hasattr(model, 'position_embedding'):
                act_o += model.position_embedding(torch.arange(T, device=device))
                for k in transformed.keys():
                    act_t[k] += model.position_embedding(torch.arange(T, device=device))

        for l in range(model.num_layers):

            act_o = model.blocks[l](act_o)	# compute activations on originals
            x = whiten(act_o, eps)

            result[l] = {}
            for k in transformed.keys():

                act_t[k] = model.blocks[l](act_t[k])			# compute the transformed activations...
                x_t = whiten(act_t[k], eps)				# ...whiten over batch dimension...
                sensitivity = F.cosine_similarity(x, x_t, dim=2)	# ...and compute cosine_sim with originals
                result[l][k] = sensitivity.mean(dim=0) # TODO: sum instead of mean for batching

        x = whiten(model(data), eps)	# same for model output
        result[l+1] = {}
        for k in transformed.keys():

            x_t = whiten(model(transformed[k].to(device)), eps)
            sensitivity = F.cosine_similarity(x, x_t, dim=1)
            result[l+1][k] = sensitivity.mean(dim=0)

    return result
