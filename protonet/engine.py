import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData
import matplotlib.pyplot as plt

'''
Function for computing the euclidian distance between two tensors 
(From: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py)
'''
def pairwise_distances_logits(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return -torch.pow(x - y, 2).sum(2)

'''
Function for computing classification accuracy
'''
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

'''
Function for computing one learn2learn task batch (source: learn2learn)
'''
def run_batch(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc

'''
Function for training model (source: learn2learn)
'''
def run_train(n_epochs=100,
              tasks_per_epoch=100, 
              shot=5, 
              train_query=1, 
              train_way=5, 
              train_loader=None,
              val_loader=None,
              optimizer=None,
              lr_scheduler=None,
              model=None,
              save_path='model_final.pth'):
    
    device = torch.device('cuda')
    model.to(device)
    best_loss = 1000

    # start training!
    for epoch in range(1, n_epochs + 1):
        init_time = time.time()
        
        model.train()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        
        for i in range(tasks_per_epoch):
            batch = next(iter(train_loader))
            loss, acc = run_batch(model,
                                   batch,
                                   train_way,
                                   shot,
                                   train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        
        print('(Train) Epoch {}/{}: loss={:.4f} acc={:.4f}'.format(
            epoch, n_epochs, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i, batch in enumerate(val_loader, 1):
            loss, acc = run_batch(model,
                                   batch,
                                   test_way,
                                   shot,
                                   test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc        
        
            
        print('(Validation) Epoch {}/{}: loss={:.4f} acc={:.4f}'.format(
            epoch, n_epochs, n_loss/loss_ctr, n_acc/loss_ctr))
        t_ = time.time() - init_time
        print(f'\033[1mEstimated epoch time\033[0m: {t_}s \n')
        print(f'\033[1mETA\033[0m: {(n_epochs - epoch) * t_ / 60}min \n')

    

'''
Function for running test on learn2learn tasks (source: learn2learn)
'''
def run_test(shot=5, 
             test_query=1, 
             test_way=5, 
             test_loader=None,
             model=None):
    
    device = torch.device('cuda')
    model.to(device)

    # start testing!
    loss_ctr = 0
    n_acc = 0
    for i, batch in enumerate(test_loader, 1):
        loss, acc = run_batch(model,
                              batch,
                              test_way,
                              shot,
                              test_query,
                              metric=pairwise_distances_logits,
                              device=device)
        loss_ctr += 1
        n_acc += acc
        print('Batch {}/{}: Batch Accuracy = {:.2f} / Total Accuracy = {:.2f}'.format(
            i, len(test_loader), acc * 100, n_acc/loss_ctr * 100))


'''
Run batch given a regular torch dataloader and a task pool
'''
def run_batch_dataloader(model, query_batch, support_batch, ways=5, shot=1, device=None):
    query_input, query_label = query_batch
    query_input = query_input.to(device)
    query_label = query_label.to(device)

    support_input, support_label = support_batch
    # print(support_input.shape)
    # grid_img = torchvision.utils.make_grid(support_input, nrow=1)
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.show()


    support_input = support_input.to(device)
    support_label = support_label.to(device)

    # Sort data samples by labels
    sort = torch.sort(support_label)
    support_input = support_input.squeeze(0)[sort.indices].squeeze(0)
    support_label = support_label.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings_query = model(query_input)
    embeddings_support = model(support_input)
    embeddings_support = embeddings_support.reshape(ways, shot, -1).mean(dim=1)

    logits = pairwise_distances_logits(embeddings_query, embeddings_support)
    loss = F.cross_entropy(logits, query_label)
    acc = accuracy(logits, query_label)

    real_labels = query_label.to('cpu').numpy()
    predicted_labels = logits.argmax(dim=1).view(query_label.shape).to('cpu').numpy()

    return loss, acc, real_labels, predicted_labels

'''
Run test given a dataloader and a task pool
'''
def run_test_dataloader(model=None, test_loader=None, task_pool=None, ways=5, shot=1):
    
    device = torch.device('cuda')
    model.to(device)

    results_dict = {'predicted': np.array([], dtype=int), 
                    'real': np.array([], dtype=int)}

    # start testing!
    loss_ctr = 0
    n_acc = 0
    for i, query_batch in enumerate(test_loader, 1):
        support_batch = task_pool.sample()
        loss, acc, real_labels, predicted_labels = run_batch_dataloader(model=model, 
                                         query_batch=query_batch, 
                                         support_batch=support_batch, 
                                         ways=ways, 
                                         shot=shot, 
                                         device=device)
        results_dict['predicted'] = np.append(results_dict['predicted'], predicted_labels)
        results_dict['real'] = np.append(results_dict['real'], real_labels)

        loss_ctr += 1
        n_acc += acc
        print('Batch {}/{}: Batch Accuracy = {:.2f} / Total Accuracy = {:.2f}'.format(
            i, len(test_loader), acc * 100, n_acc/loss_ctr * 100))
    
    return results_dict


'''
Run train given a dataloader and a task pool
'''
def run_train_dataloader(n_epochs=100, train_loader=None, val_loader=None, task_pool=None, model=None, optimizer=None, lr_scheduler=None, ways=5, shot=1, save_path='model_final.pth'):
    
    device = torch.device('cuda')
    model.to(device)
    start_time = time.time()
    time_list = []
    best_loss = 1000
    # start training!
    for epoch in range(1, n_epochs + 1):
        init_time = time.time()
        
        model.train()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        
        for i, query_batch in enumerate(train_loader, 1):
            support_batch = task_pool.sample()
            loss, acc, _, _ = run_batch_dataloader(model=model, 
                                             query_batch=query_batch, 
                                             support_batch=support_batch, 
                                             ways=ways, 
                                             shot=shot,
                                             device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        
        print('(Train) Epoch {}/{}: loss={:.4f} acc={:.4f}'.format(
            epoch, n_epochs, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i, query_batch in enumerate(val_loader, 1):
            support_batch = task_pool.sample()
            loss, acc, _, _ = run_batch_dataloader(model=model, 
                                             query_batch=query_batch, 
                                             support_batch=support_batch, 
                                             ways=ways, 
                                             shot=shot,
                                             device=device)
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        val_loss = (n_loss/loss_ctr)
        if best_loss > val_loss:
            torch.save(model.state_dict(), save_path)
            best_loss = val_loss
            print(f'Saved a new model with lower loss in epoch {epoch}!')    

        print('(Validation) Epoch {}/{}: loss={:.4f} acc={:.4f}'.format(
            epoch, n_epochs, n_loss/loss_ctr, n_acc/loss_ctr))
        t_ = time.time() - init_time
        time_list.append(t_)
        print(f'\033[1mEstimated epoch time\033[0m: {t_}s \n')
        print(f'\033[1mETA\033[0m: {(n_epochs - epoch) * t_ / 60}min \n')

    return ((time.time() - start_time) / 60), time_list

'''
Function for defining task pool
'''
def create_task_pool(dataset=None, num_tasks=100, ways=5, shot=1):
    dataset = l2l.data.MetaDataset(dataset)
    transforms = [
        NWays(dataset, ways),
        KShots(dataset, shot),
        LoadData(dataset),
    ]
    task_pool = l2l.data.TaskDataset(dataset, task_transforms=transforms, num_tasks=num_tasks)
    return task_pool

