import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple
from model import GPT, GPTConfig

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs[:, :-1]  # truncate inputs at -1 position
        targets = targets[:, 1:].contiguous().view(-1)  # shift targets
        # create a fake mask that contains False at all positions
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        optimizer.zero_grad()
        _, loss = model(inputs, mask, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

if __name__ == '__main__':
    data = [[0, 1, 2, 3, 4]]
    inputs = torch.tensor(data*100)
    targets = inputs.clone()
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    config = GPTConfig(vocab_size=5, n_layer=2, n_head=2, n_embed=16, n_hidden=32, pad_token_id=-100)
    model = GPT(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0)
    for epoch in range(1000):
        loss = train_epoch(model, train_loader, optimizer, 'cuda')
        print(f"Epoch {epoch}: loss={loss}")
