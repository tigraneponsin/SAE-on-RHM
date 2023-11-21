import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train( model, dataloader, accumulation, criterion, optimizer, scheduler):
    """
    Train the model for one epoch.
    
    Returns:
        Average loss over batches
    """
    model.train()
    optimizer.zero_grad()
    running_loss = 0.

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        outputs = model(inputs)
        loss = criterion(outputs, targets) 

        running_loss += loss.item()
        loss /= accumulation
        loss.backward()

        if ((batch_idx+1)%accumulation==0):
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()

    return running_loss / (batch_idx+1)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
