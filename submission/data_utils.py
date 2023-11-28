# Load dataset
import torch
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
from PIL import Image

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

# import pickle

# with open("./models/tokens.pkl", "wb") as f:
#     pickle.dump(dataset.tokens, f)

# with open("./models/token_to_idx.pkl", "wb") as f:
#     pickle.dump(dataset.token_to_idx, f)