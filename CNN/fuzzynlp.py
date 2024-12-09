import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define tokenizer and load IMDB dataset
tokenizer = get_tokenizer('spacy')
train_data, test_data = IMDB(split='train'), IMDB(split='test')

# Create a vocabulary from the training data
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Convert text to numerical indices
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

def label_pipeline(label):
    return 1 if label == 'pos' else 0  # 1 for positive, 0 for negative

# Example to check tokenization and vocabulary
sample_text = "The movie was fantastic!"
sample_tokens = tokenizer(sample_text)
sample_indices = text_pipeline(sample_text)
print("Sample tokens:", sample_tokens)
print("Sample indices:", sample_indices)

# Define CNN Model
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_sizes, num_filters, dropout=0.5):
        super(CNNTextClassifier, self).__init__()
        
        # Word Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (kernel_size, embedding_dim)) for kernel_size in kernel_sizes
        ])
        
        # Fully connected layer for classification
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, sequence_length, embedding_dim)
        
        conv_results = []
        for conv in self.convs:
            conv_out = conv(x)  # Shape: (batch_size, num_filters, sequence_length-kernel_size+1, 1)
            conv_out = F.relu(conv_out)  # Apply ReLU activation
            pooled = F.max_pool2d(conv_out, (conv_out.size(2), 1))  # Max pooling along the sequence dimension
            conv_results.append(pooled.squeeze(3))  # Remove the last dimension (channel dimension)
        
        out = torch.cat(conv_results, dim=1)  # Shape: (batch_size, num_filters * len(kernel_sizes))
        out = out.view(out.size(0), -1)  # Flatten the output
        
        out = self.dropout(out)
        out = self.fc(out)  # Shape: (batch_size, num_classes)
        
        return out

# Initialize the model
vocab_size = len(vocab)
embedding_dim = 100  # GloVe embedding dimension
num_classes = 1  # Binary classification (sentiment: positive/negative)
kernel_sizes = [3, 4, 5]  # Different kernel sizes for convolutions
num_filters = 128  # Number of filters (channels) in each convolution layer

model = CNNTextClassifier(vocab_size, embedding_dim, num_classes, kernel_sizes, num_filters)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

# Helper function to compute accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

# Training function
def train(model, data_iter, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for label, text in data_iter:
        optimizer.zero_grad()
        text = torch.tensor(text).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        label = torch.tensor([label_pipeline(l) for l in label]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Forward pass
        predictions = model(text).squeeze(1)  # Shape: (batch_size, 1) -> squeeze to (batch_size,)
        
        # Compute loss
        loss = criterion(predictions, label.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()

        acc = binary_accuracy(predictions, label)
        epoch_loss += loss.item()
        epoch_accuracy += acc.item()

    return epoch_loss / len(data_iter), epoch_accuracy / len(data_iter)

# Evaluation function
def evaluate(model, data_iter, criterion):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():
        for label, text in data_iter:
            text = torch.tensor(text).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            label = torch.tensor([label_pipeline(l) for l in label]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            predictions = model(text).squeeze(1)
            loss = criterion(predictions, label.float())
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_accuracy += acc.item()

    return epoch_loss / len(data_iter), epoch_accuracy / len(data_iter)

# Prepare the data iterators
train_iter = torch.utils.data.DataLoader(train_data, batch_size=64, collate_fn=lambda batch: tuple(zip(*batch)))
test_iter = torch.utils.data.DataLoader(test_data, batch_size=64, collate_fn=lambda batch: tuple(zip(*batch)))

# Training loop
N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_iter, criterion)
    
    print(f'Epoch {epoch+1}/{N_EPOCHS}')
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

# Example usage for prediction:
def predict_sentiment(model, sentence):
    model.eval()
    tokenized = tokenizer(sentence)
    indexed = text_pipeline(sentence)
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Add batch dimension
    prediction = model(tensor)[0]
    return torch.round(torch.sigmoid(prediction)).item()  # Returns 0 for negative, 1 for positive

# Example usage for prediction
sentence = "The movie was fantastic!"
print(predict_sentiment(model, sentence))
