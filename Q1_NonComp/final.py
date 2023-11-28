import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from PIL import Image
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
        if hidden is None:
            output, hidden = self.lstm(input)
        else:
            output, hidden = self.lstm(input, hidden)
        output = self.output(output)

        return output, hidden
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder=None, decoder=None, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()

    def forward(self, input):
        context_vec = self.encoder(input).squeeze().to(self.device)
        prev_token = torch.tensor([self.decoder.vocab_dict["<sos>"]], device=self.device)

        input = torch.cat([context_vec.unsqueeze(0).to(self.device), self.decoder.embedding(prev_token).to(self.device)], dim=1).to(self.device)
        hidden = None

        outputs = []

        for i in range(629):
            output, hidden = self.decoder(input, hidden)
            prev_token = torch.argmax(output, dim=1)

            if prev_token.item() == self.decoder.vocab_dict["<eos>"]:
                break
            
            outputs.append(self.decoder.vocab[prev_token.item()])
            input = torch.cat((context_vec.unsqueeze(0), self.decoder.embedding(prev_token)), dim=1).to(self.device)

        return outputs
    
    
# Load dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"

def load_img(path, size = (224, 224)):
    img = transforms.ToTensor()(Image.open(path))

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    transform = transforms.Compose([transforms.Resize(size, antialias=True), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    im = transform(img).detach()
    im = 1 - im
    return im

class Img2LatexDataset(data.Dataset):
    def __init__(self, img_dir, formula_path, img_size = (224, 224), tokens = None, token_to_idx = None):
        self.data_frame = pd.read_csv(formula_path)
        self.img_dir = img_dir
        self.img_size = img_size

        if tokens is None:
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
        else:
            self.token_to_idx = token_to_idx
            self.tokens = tokens

        max_len = max([len(row.split()) for row in self.data_frame["formula"]])+2
        def indexer(row):
            index_list = [self.token_to_idx[SOS]]
            index_list.extend([self.token_to_idx.get(token, 0) for token in row.split()])
            index_list.append(self.token_to_idx[EOS])
            index_list.extend([self.token_to_idx[PAD]] * (max_len - len(index_list)))

            return index_list
        
        self.data_frame["IndexList"] = self.data_frame["formula"].apply(indexer)

    def __getitem__(self, index):
        img = load_img(self.img_dir + self.data_frame["image"][index], self.img_size)
        if img.shape == (1, 224, 244):
            img = img.repeat(3, 1, 1)
            
        return img, torch.tensor(self.data_frame["IndexList"][index], requires_grad=False)

    def __len__(self):
        return len(self.data_frame)
    
    def get_vocab(self):
        return self.token_to_idx, self.tokens

img_dir = "../data/SyntheticData/images/"
formula_dir = "../data/SyntheticData/train.csv"

dataset = Img2LatexDataset(img_dir, formula_dir)


hparams = {
    "lr" : 0.0001,
    "batch_size" : 96,
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

#Training loop

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
PAD_IDX = dataset.token_to_idx[PAD]
if device == "cuda":
    torch.cuda.empty_cache()
def remove_trailing_pads(labels):
   # Clip trailing PAD on labels
   non_pad_cols = (labels != PAD_IDX).sum(dim=0).to(device)
   non_pad_cols = non_pad_cols[non_pad_cols > 0].to(device)

   return labels[:, :len(non_pad_cols)].to(device)

loader = data.DataLoader(dataset, batch_size = enc.hp["batch_size"], shuffle = True)
print(len(loader))
model_path = "./models/model.pt"
model_backup_path = "./models/model_backup.pt"
current_params_path = "./models/current_params.txt" 

# state_dict = torch.load(model_path)
# torch.save(model(state_dict), model_backup_path)
# model.load_state_dict(state_dict)
if device == "cuda":
    model.cuda()
model.train()
print(f"LOADED MODEL to {device}")

fifty_fifty = False
teacher_forcing = True

prev_loss = 100
for epoch in range(10):
    torch.save(model.state_dict(), model_backup_path)
    if epoch == 1:
        fifty_fifty = True
        teacher_forcing = False
    if epoch == 3:
        fifty_fifty = False
        teacher_forcing = True  
    if epoch == 5:
        fifty_fifty = True
        teacher_forcing = False
    curr_loss = 0
    for bidx, batch in enumerate(loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        labels = remove_trailing_pads(labels).to(device)
        context_vec = model.encoder(images).squeeze()
        if (bidx%2 and fifty_fifty) or teacher_forcing:
            inputs = torch.cat([context_vec.unsqueeze(1).repeat(1, labels.shape[1], 1), model.decoder.embedding(labels)], dim=2).to(device)
            print(f"Running Batch {bidx}, Epoch {epoch}, Total Tokens: {labels.shape[1]}")
            output, _ = model.decoder(inputs, None)

            # output[labels == PAD_IDX] = 0
            # output = F.normalize(output, dim=2, p=1)
            output = output[:, :-1, :].to(device)

        else:
            output = torch.zeros((labels.shape[0], labels.shape[1]-1, len(dataset.tokens))).to(device)

            prev_token = torch.ones(labels.shape[0], dtype=int).to(device) * dataset.token_to_idx[SOS]
            prev_token_embed = model.decoder.embedding(prev_token)

            input = torch.cat([context_vec, prev_token_embed], dim=1).to(device)
            hidden = None

            for i in range(labels.shape[1]-1):
                output[:, i, :], hidden = model.decoder(input, hidden)
                prev_token = output[:, i, :].argmax(dim=1).to(device)
                prev_token_embed = model.decoder.embedding(prev_token)
                input = torch.cat([context_vec, prev_token_embed], dim=1).to(device)
            
        target = nn.functional.one_hot(labels[:,1:], num_classes=len(dataset.tokens)).float().to(device)
        # target[labels == PAD_IDX] = 0
        mask = (labels[:,1:] != PAD_IDX).to(device)
        
        # print(f"Output shape: {output.shape}, Labels shape: {labels.shape}, Target shape: {target.shape}")
        optimizer.zero_grad()
        loss = criterion(output.transpose(1, 2), target.transpose(1, 2))
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Layer: {name}, Mean: {param.grad.mean()}, Std: {param.grad.std()}")

        # optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
        curr_loss += loss.item()
        if bidx % 10 == 9:
            print(f"SAVING MODEL to {model_path}")
            torch.save(model.state_dict(), model_path)
            print("SAVED MODEL")
            print(f"Epoch: {epoch}, Batch: {bidx}, Loss: {loss.item()}")
            try:
                with open(current_params_path, 'w') as f:
                    f.write(f"Epoch: {epoch}, Batch: {bidx}, Loss: {loss.item()}")
            except:
                print("\n Could not write to file \n")
    print(f"AVG LOSS: {(curr_loss)/len(loader)}, Epoch: {epoch+1}")
    prev_loss = curr_loss
    
torch.save(model.state_dict(), "./models/model_5050synthfinal.pt")

# Fine Tuning

# fine tuning
handwritten_imgs = "../data/HandwrittenData/images/train/"
handwritten_labels = "../data/HandwrittenData/train_hw.csv"

handwritten_dataset = Img2LatexDataset(handwritten_imgs, handwritten_labels, tokens=model.decoder.vocab,token_to_idx=model.decoder.vocab_dict)
handwritten_loader = data.DataLoader(handwritten_dataset, batch_size=96, shuffle=True)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.encoder.parameters(), lr = 0.0001)
PAD_IDX = handwritten_dataset.token_to_idx[PAD]
if device == "cuda":
    torch.cuda.empty_cache()
def remove_trailing_pads(labels):
   # Clip trailing PAD on labels
   non_pad_cols = (labels != PAD_IDX).sum(dim=0)
   non_pad_cols = non_pad_cols[non_pad_cols > 0]

   return labels[:, :len(non_pad_cols)]

print(len(handwritten_loader))
model_path = "./models/model.pt"
model_backup_path = "./models/model_backup.pt"
current_params_path = "./models/current_params_hw.txt" 

state_dict = torch.load(model_path)
# torch.save((state_dict), model_backup_path)
model.load_state_dict(state_dict)
if device == "cuda":
    model.cuda()
model.train()
print(f"LOADED MODEL to {device}")

fifty_fifty = False
teacher_forcing = True

prev_loss = 100
for epoch in range(12):
    torch.save(model.state_dict(), model_backup_path)
    if epoch == 2:
        fifty_fifty = True
        teacher_forcing = False
    curr_loss = 0
    for bidx, batch in enumerate(handwritten_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        labels = remove_trailing_pads(labels)
        context_vec = model.encoder(images).squeeze()
        if (bidx%2 and fifty_fifty) or teacher_forcing:
            inputs = torch.cat([context_vec.unsqueeze(1).repeat(1, labels.shape[1], 1), model.decoder.embedding(labels)], dim=2)
            print(f"Running Batch {bidx}, Epoch {epoch}, Total Tokens: {labels.shape[1]}")
            output, _ = model.decoder(inputs, None)

            # output[labels == PAD_IDX] = 0
            # output = F.normalize(output, dim=2, p=1)
            output = output[:, :-1, :]

        else:
            output = torch.zeros((labels.shape[0], labels.shape[1]-1, len(handwritten_dataset.tokens))).to(device)

            prev_token = torch.ones(labels.shape[0], dtype=int).to(device) * handwritten_dataset.token_to_idx[SOS]
            prev_token_embed = model.decoder.embedding(prev_token).to(device)

            input = torch.cat([context_vec, prev_token_embed], dim=1).to(device)
            hidden = None

            for i in range(labels.shape[1]-1):
                output[:, i, :], hidden = model.decoder(input, hidden)
                prev_token = output[:, i, :].argmax(dim=1)
                prev_token_embed = model.decoder.embedding(prev_token)
                input = torch.cat([context_vec, prev_token_embed], dim=1).to(device)
            
        target = nn.functional.one_hot(labels[:,1:], num_classes=len(handwritten_dataset.tokens)).float().to(device)
        # target[labels == PAD_IDX] = 0
        mask = labels[:,1:] != PAD_IDX
        # print(f"Output shape: {output.shape}, Labels shape: {labels.shape}, Target shape: {target.shape}")
        optimizer.zero_grad()
        loss = criterion(output.transpose(1, 2), target.transpose(1, 2))
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Layer: {name}, Mean: {param.grad.mean()}, Std: {param.grad.std()}")

        # optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
        curr_loss += loss.item()
        if bidx % 10 == 3:
            print(f"SAVING MODEL to {model_path}")
            torch.save(model.state_dict(), model_path)
            print("SAVED MODEL")
            print(f"Epoch: {epoch}, Batch: {bidx}, Loss: {loss.item()}")
            try:
                with open(current_params_path, 'w') as f:
                    f.write(f"Epoch: {epoch}, Batch: {bidx}, Loss: {loss.item()}")
            except:
                print("\n Could not write to file \n")
    print(f"AVG LOSS: {(curr_loss)/len(handwritten_loader)}, Epoch: {epoch+1}")
    prev_loss = curr_loss
    
torch.save(model.state_dict(), "./models/model_tfhwfinal.pt")