'''
Script for running ProtoNet with a list of backbones and saving a log

author: Lucas Miguel Tassis 
email: lucaswfitassis@gmail.com
'''

import os
import sys
from engine import create_task_pool, run_train_dataloader, run_test_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from models import *
import numpy as np
from tsne import *

def generate_tSNE(ways=5, shot=1, path_data=None):

    # create output folder for saving results
    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    # define backbones that will be trained
    models = [ResNet50(), MobileNetv2(), VGG16(), DenseNet121(), EfficientNetB4()]
    model_name = ['ResNet50', 'MobileNetv2', 'VGG16', 'DenseNet121', 'EfficientNetB4']

        
    val_transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = torchvision.datasets.ImageFolder(root=path_data + 'test/', transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for model, name in zip(models, model_name):
        trained_model = model
        trained_model.load_state_dict(torch.load(f'/content/drive/MyDrive/pg/output_ways={ways}_shot={shot}/model_{name}.pth'))
        trained_model.eval()

        embeddings, labels = generate_embeddings(test_loader, model)
        vis_tSNE(embeddings=embeddings, labels=labels, ways=ways, shot=shot, backbone=name)


def run_protonet_models(ways=5, shot=1, path_data=None):
    
    # define backbones that will be trained
    models = [ResNet50(), MobileNetv2(), VGG16(), DenseNet121(), EfficientNetB4()]
    model_name = ['ResNet50', 'MobileNetv2', 'VGG16', 'DenseNet121', 'EfficientNetB4']
    # models = [MobileNetv2()]
    # model_name = ['MobileNetv2']

    # create output folder for saving results
    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    # create output file
    log = open('./output/log.txt', 'w')

    # dataset transforms
    train_transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    val_transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    # define datasets and loaders
    train_dataset = torchvision.datasets.ImageFolder(root=path_data + 'train/', transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = torchvision.datasets.ImageFolder(root=path_data + 'val/', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_dataset = torchvision.datasets.ImageFolder(root=path_data + 'test/', transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    eval_dataset = torchvision.datasets.ImageFolder(root=path_data + 'train/', transform=val_transforms)
    
    # it might take a while to define the task pool
    print('Creating tasks... this may take a while...')
    train_task_pool = create_task_pool(dataset=train_dataset, num_tasks=-1, ways=ways, shot=shot)
    eval_task_pool = create_task_pool(dataset=eval_dataset, num_tasks=-1, ways=ways, shot=shot)


    for model, name in zip(models, model_name):
        print(f'Starting to training on model: {name}')

        # define optimizer and lr_scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # run training
        _, epoch_list = run_train_dataloader(n_epochs=100, 
                     train_loader=train_loader, 
                     val_loader=val_loader, 
                     task_pool=train_task_pool, 
                     model=model,
                     optimizer=optimizer, 
                     lr_scheduler=lr_scheduler, 
                     ways=ways, 
                     shot=shot, 
                     save_path=f'./output/model_{name}.pth')

        

        # run testing
        trained_model = model
        trained_model.load_state_dict(torch.load(f'/content/drive/MyDrive/pg/output_ways=5_shot=1/model_{name}.pth'))
        trained_model.eval()

        results_dict = run_test_dataloader(model=trained_model, 
                    test_loader=test_loader, 
                    task_pool=eval_task_pool, 
                    ways=ways, 
                    shot=shot)

        # save log
        y_true = results_dict['real']
        y_pred = results_dict['predicted']
        
        cm = confusion_matrix(results_dict['real'], results_dict['predicted'], normalize='true')
        df_cm = pd.DataFrame(cm, index = [i for i in ['Healthy', 'Miner', 'Rust', 'Phoma', 'Cercospora']],
                          columns = [i for i in ['Healthy', 'Miner', 'Rust', 'Phoma', 'Cercospora']])
        plt.figure(figsize = (36,27))
        sn.set(font_scale=5)
        sn.heatmap(df_cm, annot=True, cmap="Blues", square=True, vmin=0, vmax=1)
        plt.yticks(np.arange(5) + 0.5,('Healthy', 'Miner', 'Rust', 'Phoma', 'Cercospora'), rotation=0, va="center")
        plt.ylabel('True Label', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.title(f'Confusion Matrix: {name}', fontweight='bold')
        plt.savefig(f'./output/cm_{name}.png')
        plt.close()

        accuracy = round(accuracy_score(y_true, y_pred) * 100,2)
        precision = round(precision_score(y_true, y_pred, average='macro') * 100,2)
        recall = round(recall_score(y_true, y_pred, average='macro') * 100,2)
        f1 = round(f1_score(y_true, y_pred, average='macro') * 100,2)
        
        log.write(f'--- Results for model {name} ---\n')
        log.write(f'Accuracy Score:{accuracy}\n')
        log.write(f'Precision Score: {precision}\n')
        log.write(f'Recall Score: {recall}\n')
        log.write(f'F1 Score: {f1}\n')
        log.write(f'Average epoch time: {round((sum(epoch_list)/len(epoch_list)), 2)}\n')
        log.write(f'--------------------------------')
        log.write('\n\n')
        log.flush()

        print(f'Finished training on model: {name}')
        torch.cuda.empty_cache()
    log.close()
