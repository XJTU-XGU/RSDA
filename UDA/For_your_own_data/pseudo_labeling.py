import os
import numpy as np
import torch
import network
import pre_process as prep
from torch.utils.data import DataLoader
from pre_process import ImageList
from mixed_gaussian_uniform import *
import tqdm


###
def sample_weighting(features,labels,pseu_labels,n_class):
    features = features.numpy()
    labels = labels.numpy()
    pseu_labels = pseu_labels.numpy()

    id = np.arange(len(features))
    sort_index = np.argsort(pseu_labels)
    clust_features = features[sort_index]
    clust_pseu_labels = pseu_labels[sort_index]
    clust_labels = labels[sort_index]
    clust_id = id[sort_index]

    weighted_id = np.empty([0], dtype=int)
    weighted_pseu_label = np.empty([0], dtype=int)
    weights = np.empty([0])
    sigmas=np.empty([0])
    acc = 0
    length = 0
    for i in range(n_class):
        class_feature = clust_features[clust_pseu_labels == i]
        class_label = clust_labels[clust_pseu_labels == i]
        class_id = clust_id[clust_pseu_labels == i]
        if len(class_id) == 0:
            continue
        class_mean = np.mean(class_feature, axis=0)
        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-10)
        R=np.linalg.norm(class_feature,axis=1)[0]
        class_dist=np.arccos(np.sum(class_feature / R * class_mean.reshape(-1, 256), axis=1))
        class_dist = class_dist - np.min(class_dist)
        class_dist[2 * np.arange(len(class_dist) // 2)] = -1 * class_dist[2 * np.arange(len(class_dist) // 2)]
        weight, pi, sigma = gauss_unif(class_dist.reshape(-1, 1))

        label_select = class_label[weight > 0.5]
        acc_ = np.mean(label_select == i)
        acc += acc_ * len(label_select)
        length += len(label_select)
        print(i, acc_, acc, len(label_select), pi, sigma)

        weights = np.hstack((weights, weight))
        weighted_id = np.hstack((weighted_id, class_id))
        weighted_pseu_label = np.hstack((weighted_pseu_label, np.ones_like(class_id, dtype=int) * i))
        sigmas = np.hstack((sigmas, np.ones_like(class_id) * sigma))

    print(acc/length,length)
    return weighted_id,weighted_pseu_label,weights,sigmas


def make_list(id,pseu_label,weights,sigmas,list_path,save_path):
    lists = open(list_path).readlines()
    labeled_list = [lists[id[i]].split(' ')[0] + ' ' + str(pseu_label[i]) + ' '
                    + str(weights[i]) +' ' + str(sigmas[i])  for i in range(len(id))]

    fw = open(save_path, 'w')
    for l in labeled_list:
        fw.write(l)
        fw.write('\n')

def make_new_list(list_path,source,target,n_class,iter_times=0):
    save_path = 'new_list/' + source + '_' + target + '_list.txt'
    if not os.path.exists('new_list'):
        os.mkdir('new_list')
    transform = prep.image_test(**{"resize_size": 256, "crop_size": 224})
    dsets = ImageList(open(list_path).readlines(), transform=transform)
    dloader = DataLoader(dsets, batch_size=72, shuffle=False, num_workers=4, drop_last=False)

    if iter_times==0:
        model = torch.load('save/init_model/' + source + '_' + target + '.pkl')
    if iter_times>0:
        model = torch.load('save/rsda_model/' + source + '_' + target + '.pkl')

    features = torch.Tensor([])
    labels = torch.LongTensor([])
    pseu_labels = torch.LongTensor([])
    with torch.no_grad():
        for data in tqdm.tqdm(dloader):
            input = data[0]
            label = data[1]
            input = input.cuda()
            feature, outputs = model(input)
            features = torch.cat([features, feature.cpu()], dim=0)
            labels = torch.cat([labels, label], dim=0)
            pseu_labels = torch.cat([pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    weighted_id, weighted_pseu_label, weights, sigmas= sample_weighting(features, labels, pseu_labels,n_class)
    make_list(weighted_id,weighted_pseu_label,weights,sigmas, list_path, save_path)





