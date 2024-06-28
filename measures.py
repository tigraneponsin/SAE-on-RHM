import torch
import torch.nn as nn
import torch.nn.functional as F

def test( model, dataloader):
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

            outputs = model(inputs)
            _, predictions = outputs.max(1)

            loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    return loss / total, 1.0 * correct / total
