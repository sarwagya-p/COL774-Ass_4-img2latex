import torch
import torch.nn as nn
import torch.nn.functional as F
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
    model = load_model("./models/part1a.pt")
    dataset_dir = sys.argv[1]

    preds_hw = pd.read_csv(dataset_dir+"/sample_sub.csv")
    pred_sy = pd.read_csv(dataset_dir+"/SyntheticData/test.csv")

    preds_hw["formula"] = preds_hw["image"].apply(lambda x: read_and_pred(dataset_dir+"/"+x, model))