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
    model = load_model("./models/part1b.pt")
    dataset_dir = sys.argv[1]

    hw_dir = dataset_dir+"/sample_sub.csv"
    sy_dir = dataset_dir+"/SyntheticData/test.csv"

    preds_hw = pd.read_csv(hw_dir)
    pred_sy = pd.read_csv(sy_dir)

    preds_hw["formula"] = preds_hw["image"].apply(lambda x: read_and_pred(hw_dir+"/"+x, model))
    preds_hw["formula"] = preds_hw["image"].apply(lambda x: read_and_pred(sy_dir+"/"+x, model))

    preds_hw.to_csv(hw_dir)
    pred_sy.to_csv(sy_dir)