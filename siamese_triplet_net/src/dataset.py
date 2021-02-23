from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from learn2learn.data import MetaDataset
import random
from PIL import Image
import torch

class SiameseDataset(Dataset):
    def __init__(self, root=None, transforms=None):

        img_folder = ImageFolder(root=root)
        meta_data = MetaDataset(img_folder)
        
        self.img_list = img_folder.imgs
        self.labels_to_indices = meta_data.labels_to_indices
        self.indices_to_labels = meta_data.indices_to_labels
        self.labels = [0, 1, 2, 3, 4] # labels in dataset
        self.transforms = transforms

    def __getitem__(self, index):
        
        img_anchor = self.img_list[index][0]
        label_anchor = self.img_list[index][1]
        rand = random.randint(0, 1)

        # get a positive sample
        if rand == 0:
            idx = random.choice(self.labels_to_indices[label_anchor])            
            img_aux = self.img_list[idx][0]
            label_aux = self.img_list[idx][1]
            label = torch.FloatTensor([1]) # similar image, label = 1
        
        # get a negative sample
        else:
            aux_class = random.choice(list(set(self.labels) - set([label_anchor]))) # select a random label from the labels list
            idx = random.choice(self.labels_to_indices[aux_class])
            img_aux = self.img_list[idx][0]
            label_aux = self.img_list[idx][1]
            label = torch.FloatTensor([0]) # different image, label = 0

        img_anchor = Image.open(img_anchor)
        img_aux = Image.open(img_aux)

        if self.transforms is not None:
            img_anchor = self.transforms(img_anchor)
            img_aux = self.transforms(img_aux)

        return (img_anchor, img_aux), label


    def __len__(self):
        return len(self.img_list)


class TripletDataset(Dataset):
    def __init__(self, root=None, transforms=None):

        img_folder = ImageFolder(root=root)
        meta_data = MetaDataset(img_folder)
        
        self.img_list = img_folder.imgs
        self.labels_to_indices = meta_data.labels_to_indices
        self.indices_to_labels = meta_data.indices_to_labels
        self.labels = [0, 1, 2, 3, 4] # labels in dataset
        self.transforms = transforms

    def __getitem__(self, index):
        
        img_anchor = self.img_list[index][0]
        label_anchor = self.img_list[index][1]

        # get a positive sample
        idx_positive = random.choice(self.labels_to_indices[label_anchor])            
        img_positive = self.img_list[idx_positive][0]
        label_positive = self.img_list[idx_positive][1]
        
        # get a negative sample
        aux_class = random.choice(list(set(self.labels) - set([label_anchor]))) # select a random label from the labels list
        idx_negative = random.choice(self.labels_to_indices[aux_class])
        img_negative = self.img_list[idx_negative][0]
        label_negative = self.img_list[idx_negative][1]

        img_anchor = Image.open(img_anchor)
        img_positive = Image.open(img_positive)
        img_negative = Image.open(img_negative)

        if self.transforms is not None:
            img_anchor = self.transforms(img_anchor)
            img_positive = self.transforms(img_positive)
            img_negative = self.transforms(img_negative)

        return (img_anchor, img_positive, img_negative), []


    def __len__(self):
        return len(self.img_list)



if __name__ == '__main__':

    root_path = './dataset/train'

    # dataset = SiameseDataset(root=root_path)
    # print(dataset[random.randint(0, 1200)])
    # print(len(dataset))

    dataset = TripletDataset(root=root_path)
    print(len(dataset))


