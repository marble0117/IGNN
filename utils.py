import torch

def devide_dataset(dataset, num_train_per_class, num_val, num_test):
    num_class = dataset.num_classes
    data = dataset[0]
    # return train_mask, val_mask, test_mask

def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]