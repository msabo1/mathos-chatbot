from torch.utils.data import Dataset


class ChatbotDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    
    def __len__(self):
        return len(self.X)
