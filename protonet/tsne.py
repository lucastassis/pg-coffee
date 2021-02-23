# from: https://github.com/avilash/pytorch-siamese-triplet/blob/master/tsne.py

import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable
import os
import pandas as pd
import seaborn as sns
import torch

def generate_embeddings(data_loader, model):
    with torch.no_grad():
        device = 'cuda'
        model.eval()
        model.to(device)
        labels = None
        embeddings = None
        for batch_idx, data in tqdm(enumerate(data_loader)):
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            batch_imgs = Variable(batch_imgs.to(device))
            batch_E = model(batch_imgs)
            batch_E = batch_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, batch_E), axis=0) if embeddings is not None else batch_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels

def vis_tSNE(embeddings, labels, ways=5, shot=1, backbone='ConvNet'):
    num_samples = embeddings.shape[0]
    X_embedded = TSNE(n_components=2).fit_transform(embeddings[0:num_samples, :])
    plt.figure(figsize=(16, 16))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels_name = ['Healthy', 'Miner', 'Rust', 'Phoma', 'Cercospora']
    for i in range(5):
        inds = np.where(labels==i)[0]
        plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=.8, color=colors[i], s=200)
    # plt.title(f't-SNE', fontweight='bold', fontsize=24)
    plt.legend(labels_name, fontsize=30)
    plt.savefig(f'./output/tsne_{backbone}.png')