import torch
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
from PIL import Image

def load_img(path, size = (224, 224)):
    img = Image.open(path)
    img = img.resize(size)

    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return transform(img)

class Img2LatexDataset(data.Dataset):
    def __init__(self, img_dir, formula_path, img_size = (224, 224)):
        self.data_frame = pd.read_csv(formula_path)
        
        self.vectors = torch.tensor((len(self.data_frame), 3, img_size[0], img_size[1]))

        for i in range(len(self.data_frame)):
            img = load_img(img_dir + self.data_frame["image"][i])
            self.vectors[i] = img
        
        # Normalize images
        self.mean = torch.mean(self.vectors, dim = 0)
        self.std = torch.std(self.vectors, dim = 0)

        self.vectors = (self.vectors - self.mean) / self.std

        self.transform = transforms.Compose([transforms.Resize(img_size), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(self.mean, self.std)])

    def __getitem__(self, index):
        return self.vectors[index], self.data_frame["formula"][index].split()

    def __len__(self):
        return len(self.data_frame)
    
    def get_transform(self):
        return self.transform
    

if __name__ == "__main__":
    img_dir = "../data/SyntheticData/images/"
    formula_dir = "../data/SyntheticData/train.csv"

    dataset = Img2LatexDataset(img_dir, formula_dir)

    dataset[0]