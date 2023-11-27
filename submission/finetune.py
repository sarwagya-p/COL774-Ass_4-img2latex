from EncoderDecoder import EncoderCNN, DecoderRNN, EncoderDecoder
from data_utils import Img2LatexDataset
from train_model import *
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

handwritten_imgs = "../data/HandwrittenData/images/"
handwritten_labels = "../data/HandwrittenData/train_hw.csv"

handwritten_dataset = Img2LatexDataset(handwritten_imgs, handwritten_labels)
handwritten_loader = DataLoader(handwritten_dataset, batch_size=64, shuffle=True)

model = EncoderDecoder().to(device)

state_dict = torch.load(model_path)
torch.save((state_dict), model_backup_path)
model.load_state_dict(state_dict)
model.train()
print(f"LOADED MODEL to {device}")

enc_optimizer = torch.optim.Adam(model.enc.parameters(), lr=0.001)
dec_optimizer = torch.optim.Adam(model.dec.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss(reduction="none")

train_model(model, criterion, enc_optimizer, handwritten_loader, dec_optimizer, True)