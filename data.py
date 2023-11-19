import torch
import torch.utils.data as data
import pandas as pd

class Img2LatexDatasset(data.Dataset):
    def __init__(self, img_dir, formula_path, transform=None):
        self.img_to_formula = pd.read_csv(formula_path)
        self.img_to_formula.set_index("image")
        self.transform = transform

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass