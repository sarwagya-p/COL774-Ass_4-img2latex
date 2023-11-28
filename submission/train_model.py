import torch
import torch.nn as nn
import torch.nn.functional as F
from EncoderDecoder import EncoderCNN, DecoderRNN, EncoderDecoder
from data_utils import Img2LatexDataset
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()

model_path = "./models/model.pt"
model_backup_path = "./models/model_backup.pt"
current_params_path = "./models/current_params.txt" 

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"

def load_model(model_path, eval = True):
    hparams = {
        "lr" : 0.001,
        "batch_size" : 64,
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

    with open("./models/vocab.pkl", "rb") as f:
        tokens, tokens_to_idx = pickle.load(f)
        
    dec = DecoderRNN(tokens, tokens_to_idx, 512, 512).to(device)

    model = EncoderDecoder(enc, dec).to(device)

    state_dict = torch.load(model_path, map_location=torch.device(device))
    torch.save((state_dict), model_backup_path)
    model.load_state_dict(state_dict)

    if eval:
        model.eval()
    else:
        model.train()
        
    print(f"LOADED MODEL to {device}")
    global PAD_IDX
    PAD_IDX = model.decoder.vocab_dict[PAD]
    return model

def remove_trailing_pads(labels):
   # Clip trailing PAD on labels
   non_pad_cols = (labels != PAD_IDX).sum(dim=0)
   non_pad_cols = non_pad_cols[non_pad_cols > 0]

   return labels[:, :len(non_pad_cols)]


def train_model(model, criterion, optimizer, loader, frozen_optim = None, fifty_fifty = False, teacher_enforcing = False):
    prev_loss = 100
    for epoch in range(100):
        curr_loss = 0
        for bidx, batch in enumerate(loader):
            print(f"Running Batch {bidx}, Epoch {epoch}")
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            labels = remove_trailing_pads(labels)
            
            if fifty_fifty and batch %2 == 0 or teacher_enforcing:
                context_vec = model.encoder(images).squeeze()

                output = torch.zeros((labels.shape[0], labels.shape[1]-1, len(model.decoder.vocab))).to(device)

                prev_token = torch.ones(labels.shape[0], dtype=int).to(device) * model.decoder.vocab_dict[SOS]
                prev_token_embed = model.decoder.embedding(prev_token).to(device)

                input = torch.cat([context_vec, prev_token_embed], dim=1).to(device)
                hidden = None

                for i in range(labels.shape[1]-1):
                    output[:, i, :], hidden = model.decoder(input, hidden)
                    prev_token = output[:, i, :].argmax(dim=1)
                    prev_token_embed = model.decoder.embedding(prev_token)
                    input = torch.cat([context_vec, prev_token_embed], dim=1).to(device)

            else:
                inputs = torch.cat([context_vec.unsqueeze(1).repeat(1, labels.shape[1], 1), model.decoder.embedding(labels)], dim=2)
                output, _ = model.decoder(inputs, None)
                output = output[:, :-1, :]

            target = nn.functional.one_hot(labels[:,1:], num_classes=len(model.decoder.vocab)).float().to(device)
            optimizer.zero_grad()

            if frozen_optim is not None:
                frozen_optim.zero_grad()
                
            loss = criterion(output.transpose(1, 2), target.transpose(1, 2))
            loss = loss[labels[:,1:] != PAD_IDX].mean()
            loss.backward(retain_graph=True)
            optimizer.step()

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