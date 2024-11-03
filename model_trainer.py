import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('./train2.json') as f:
    data = json.load(f)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = get_tokenizer("basic_english")
        self.tokenized_data = [(self.tokenizer(item['title']), torch.tensor(item['label'], dtype=torch.float32)) for item in data]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

train_dataset = CustomDataset(data)

# Function to yield tokens from the dataset for vocabulary building

# Build the vocabulary from the tokenized data
vocab = build_vocab_from_iterator((tokens for tokens, _ in train_dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def collate_batch(batch):
    text_list, label_list, offsets = [], [], [0]
    for tokens, _label in batch:
        processed_text = torch.tensor(vocab(tokens), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(_label)
        offsets.append(processed_text.size(0))

    # Pad and concatenate text
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    label_list = torch.stack(label_list)
    
    return text_list, offsets, label_list

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return torch.sigmoid(self.fc(embedded))

# Model parameters
vocab_size = len(vocab)
embed_dim = 128
num_labels = len(train_dataset[0][1])  # Length of the label vector

model = TextClassificationModel(vocab_size, embed_dim, num_labels).to(device)


criterion = nn.BCELoss()  # Binary cross-entropy for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.009)
def train(dataloader):
    model.train()
    total_loss = 0
    for text, offsets, labels in tqdm(dataloader, desc="Training Progress"):
        text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Example training loop
for epoch in range(7):
    loss = train(train_loader)
    print(f'Epoch {epoch+1}, Loss: {loss}')

    torch.save(model.state_dict(), 'text_classification_model.pth')
