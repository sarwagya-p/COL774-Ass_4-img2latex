import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import Img2LatexDataset
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()

with open("model/tokens.pkl", "rb") as f:
    tokens = pickle.load(f)

with open("model/token_to_idx.pkl", "rb") as f:
    token_to_idx = pickle.load(f)

model_path = "./models/model.pt"
model_backup_path = "./models/model_backup.pt"
current_params_path = "./models/current_params.txt" 

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"

PAD_IDX = token_to_idx[PAD]


def remove_trailing_pads(labels):
   # Clip trailing PAD on labels
   non_pad_cols = (labels != PAD_IDX).sum(dim=0)
   non_pad_cols = non_pad_cols[non_pad_cols > 0]

   return labels[:, :len(non_pad_cols)]


def train_model(model, criterion, optimizer, loader, frozen_optim = None, teacher_enforcing = False):
    prev_loss = 100
    for epoch in range(100):
        curr_loss = 0
        for bidx, batch in enumerate(loader):
            print(f"Running Batch {bidx}, Epoch {epoch}")
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            labels = remove_trailing_pads(labels)
            
            if not teacher_enforcing:
                context_vec = model.encoder(images).squeeze()

                output = torch.zeros((labels.shape[0], labels.shape[1]-1, len(tokens))).to(device)

                prev_token = torch.ones(labels.shape[0], dtype=int).to(device) * token_to_idx[SOS]
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

            target = nn.functional.one_hot(labels[:,1:], num_classes=len(tokens)).float().to(device)
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