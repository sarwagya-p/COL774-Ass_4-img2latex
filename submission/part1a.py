import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EncoderDecoder import EncoderCNN, DecoderRNN, EncoderDecoder
from data_utils import Img2LatexDataset, load_img
from train_model import *
import pickle
import sys
import pandas as pd
import os

with open("./models/vocab.pkl", "rb") as f:
    tokens, token_to_idx = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
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
    dec = DecoderRNN(tokens, token_to_idx, 512, 512).to(device)

    model = EncoderDecoder(enc, dec).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    PAD_IDX = token_to_idx[PAD]
    if device == "cuda":
        torch.cuda.empty_cache()

    main_dir = sys.argv[1]
    train_imgs = os.path.join(main_dir, "./SyntheticData/images/")
    train_csv = os.path.join(main_dir, "./SyntheticData/train.csv")

    dataset = Img2LatexDataset(train_imgs, train_csv, tokens, token_to_idx)
    loader = DataLoader(dataset, batch_size = hparams["batch_size"], shuffle = True, num_workers = 4)
    
    train_model(model, criterion, optimizer, loader, max_epochs=2)
    train_model(model, criterion, optimizer, loader, fifty_fifty=True, max_epochs=4)
    train_model(model, criterion, optimizer, loader, teacher_enforcing=False, max_epochs=2)

    run_model(model, main_dir)