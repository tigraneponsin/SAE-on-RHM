import torch

def test( model, dataloader):
    """
    Test the accuracy of model in predicting the output of data from dataloader.
    
    Returns:
        Classification accuracy of model in [0,1].
    """
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:

            outputs = model( inputs)
            _, predictions = outputs.max(1)

            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    return 1.0 * correct / total
