# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, layers, hparams):
        '''
        Args:
            layers: Description of all layers in the Encoder: [(layer_type, {layer_params})]
                - layer types - ['conv1d', 'conv2d', 'maxpool1d', 'maxpool2d', 'avgpool2d', 'avgpool2d', 'linear', 'dropout']
                - layer_params - dict of parameters for the layer

            hparams: Hyperparameters for the model
        '''
        super(EncoderCNN, self).__init__()
        self.hp = hparams
        self.layers = nn.ModuleList()

        for layer_type, layer_params in layers:
            if layer_type == 'conv1d':
                self.layers.append(nn.Conv1d(**layer_params))
            elif layer_type == 'conv2d':
                self.layers.append(nn.Conv2d(**layer_params))
            elif layer_type == 'maxpool1d':
                self.layers.append(nn.MaxPool1d(**layer_params))
            elif layer_type == 'maxpool2d':
                self.layers.append(nn.MaxPool2d(**layer_params))
            elif layer_type == 'avgpool1d':
                self.layers.append(nn.AvgPool1d(**layer_params))
            elif layer_type == 'avgpool2d':
                self.layers.append(nn.AvgPool2d(**layer_params))
            elif layer_type == 'linear':
                self.layers.append(nn.Linear(**layer_params))
            elif layer_type == 'dropout':
                self.layers.append(nn.Dropout(**layer_params))
            else:
                raise ValueError(f'Invalid layer type: {layer_type}')

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
class DecoderRNN(nn.Module):
    def __init__(self, vocab, vocab_dict, input_size, embedding_size):
        super(DecoderRNN, self).__init__()
        '''
        Args:
            vocabulary_size: Size of the vocabulary
            embedding_size: Size of the embedding vector
        '''

        self.vocab = vocab
        self.vocab_dict = vocab_dict

        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(input_size+embedding_size, embedding_size, batch_first=True)
        self.output = nn.Linear(embedding_size, len(vocab))

    def forward(self, input, hidden):
        '''
        Args:
            input: Input to the decoder
            hidden: Hidden state of the previous time step
        '''
        # prev_embed = self.embedding(prev_tokens)
        # concated_inp = torch.cat((input, prev_embed), dim=1)

        output, hidden = self.lstm(input, hidden)
        output = self.output(output)


        return output, hidden
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

# %%
# Load dataset
import torch.utils.data as data
from torchvision import transforms
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"

def load_img(path, size = (224, 224)):
    img = (Image.open(path))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size, antialias=True), transforms.Normalize(0, 255)])
    return transform(img).detach()

class Img2LatexDataset(data.Dataset):
    def __init__(self, img_dir, formula_path, img_size = (224, 224)):
        self.data_frame = pd.read_csv(formula_path)
        self.img_dir = img_dir
        self.img_size = img_size

        self.token_to_idx = {}
        self.tokens = []

        for row in self.data_frame["formula"]:
            row = row.split()

            for token in row:
                if token not in self.token_to_idx:
                    self.token_to_idx[token] = len(self.token_to_idx)
                    self.tokens.append(token)
        
        for special_token in [SOS, EOS, PAD]:
            self.token_to_idx[special_token] = len(self.token_to_idx)
            self.tokens.append(special_token)

        max_len = max([len(row.split()) for row in self.data_frame["formula"]])+2
        def indexer(row):
            index_list = [self.token_to_idx[SOS]]
            index_list.extend([self.token_to_idx[token] for token in row.split()])
            index_list.append(self.token_to_idx[EOS])
            index_list.extend([self.token_to_idx[PAD]] * (max_len - len(index_list)))

            return index_list
        
        self.data_frame["IndexList"] = self.data_frame["formula"].apply(indexer)

    def __getitem__(self, index):
        img = load_img(self.img_dir + self.data_frame["image"][index], self.img_size)
        return img, torch.tensor(self.data_frame["IndexList"][index], requires_grad=False)

    def __len__(self):
        return len(self.data_frame)
    
    def get_vocab(self):
        return self.token_to_idx, self.tokens

img_dir = "../data/SyntheticData/images/"
formula_dir = "../data/SyntheticData/train.csv"

dataset = Img2LatexDataset(img_dir, formula_dir)


# %%
hparams = {
    "lr" : 0.001,
    "batch_size" : len(dataset),
    "epochs" : 10
}

channel_seq = [3, 32, 64, 128, 256, 512]
num_conv_pool = 5

enc_layers = []

for i in range(num_conv_pool):
    enc_layers.append(('conv2d', {'in_channels': channel_seq[i], 'out_channels': channel_seq[i+1], 'kernel_size': 5}))
    enc_layers.append(('maxpool2d', {'kernel_size': 2}))

enc_layers.append(('avgpool2d', {'kernel_size': (3,3)}))

enc = EncoderCNN(enc_layers, hparams).to(device)
dec = DecoderRNN(dataset.tokens, dataset.token_to_idx, 512, 512).to(device)

model = EncoderDecoder(enc, dec).to(device)

# %%
# print(f"Longest formula in training: {max([len(formula) for formula in dataset.data_frame['IndexList']])}")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
PAD_IDX = dataset.token_to_idx[PAD]

def remove_trailing_pads(labels):
   # Clip trailing PAD on labels
   non_pad_cols = (labels != PAD_IDX).sum(dim=0)
   non_pad_cols = non_pad_cols[non_pad_cols > 0]

   return labels[:, :len(non_pad_cols)]

loader = data.DataLoader(dataset, batch_size = enc.hp["batch_size"], shuffle = False)

batch = next(iter(loader))
images, labels = batch
images = images.to(device)
labels = labels.to(device)
labels = remove_trailing_pads(labels)
context_vec = model.encoder(images).squeeze()
inputs = torch.cat([context_vec.unsqueeze(1).repeat(1, labels.shape[1], 1), model.decoder.embedding(labels)], dim=2)


model_path = "./models/model.pt"
current_params_path = "./models/current_params.txt"

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.train()
print(f"LOADED MODEL to {device}")

prev_loss = 1000000
for epoch in range(1000):
    curr_loss = 0
    # for bidx, batch in enumerate(loader):
    for bidx in range(1):
        # images, labels = batch
        # images = images.to(device)
        # labels = labels.to(device)
        
        # labels = remove_trailing_pads(labels)
        # context_vec = model.encoder(images).squeeze()

        # inputs = torch.cat([context_vec.unsqueeze(1).repeat(1, labels.shape[1], 1), model.decoder.embedding(labels)], dim=2)
        print(f"Running Batch {bidx}, Epoch {epoch}, Total Tokens: {labels.shape[1]}")
        output, _ = model.decoder(inputs, None)

        output[labels == PAD_IDX] = 0
        output = F.normalize(output, dim=2, p=1)
        optimizer.zero_grad()

        target = nn.functional.one_hot(labels, num_classes=len(dataset.tokens)).float().to(device)
        # print(f"Output shape: {output.shape}, Labels shape: {labels.shape}, Target shape: {target.shape}")

        loss = criterion(output.transpose(1, 2), target.transpose(1, 2))
        loss.backward(retain_graph=True)
        optimizer.step()

        # optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
        curr_loss += loss.item()
        if bidx % 10 == 0:
            print(f"SAVING MODEL to {model_path}")
            torch.save(model.state_dict(), model_path)
            print("SAVED MODEL")
            print(f"Epoch: {epoch}, Batch: {bidx}, Loss: {loss.item()}")
            try:
                with open(current_params_path, 'w') as f:
                    f.write(f"Epoch: {epoch}, Batch: {bidx}, Loss: {loss.item()}")
            except:
                print("\n Could not write to file \n")
    print(prev_loss - curr_loss)
    prev_loss = curr_loss
        

# %%



