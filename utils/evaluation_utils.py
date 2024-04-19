import torch

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            output = model(data)
            loss = torch.mean((output - data.y) ** 2)
            total_loss += loss.item()
    return total_loss / len(data_loader)