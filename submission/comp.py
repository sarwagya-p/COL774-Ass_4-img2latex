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

def read_and_pred(img_path, model):
    img = load_img(img_path)
    pred = model(img)

    return pred

if __name__ == "__main__":
    main_dir = sys.argv[1]
    handwritten_imgs = os.join(main_dir, "/HandwrittenData/images/")
    handwritten_labels = os.path.join(main_dir, "/HandwrittenData/train_hw.csv")

    handwritten_dataset = Img2LatexDataset(handwritten_imgs, handwritten_labels)
    handwritten_loader = DataLoader(handwritten_dataset, batch_size=96, shuffle=True)

    model = EncoderDecoder().to(device)

    state_dict = torch.load(model_path)
    torch.save((state_dict), model_backup_path)
    model.load_state_dict(state_dict)
    model.train()
    print(f"LOADED MODEL to {device}")

    enc_optimizer = torch.optim.Adam(model.enc.parameters(), lr=0.001)
    dec_optimizer = torch.optim.Adam(model.dec.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss(reduction="none")

    train_model(model, criterion, enc_optimizer, handwritten_loader, dec_optimizer, max_epochs=2, model_path="./models/part1b.pt")
    train_model(model, criterion, enc_optimizer, handwritten_loader, dec_optimizer, fifty_fifty=True, max_epochs=2, model_path="./models/part1b.pt")
    train_model(model, criterion, enc_optimizer, handwritten_loader, dec_optimizer, teacher_enforcing=False, max_epochs=2, model_path="./models/part1b.pt")

    run_model(model, main_dir)