import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import ImageList,image_test
from gaussian_uniform.EM_for_gaussian_uniform import gauss_unif
import os

def sample_weighting(features,labels,pseu_labels,num_class=31):
    features = features.numpy()
    labels = labels.numpy()
    pseu_labels = pseu_labels.numpy()

    id = np.arange(len(features))
    sort_index = np.argsort(pseu_labels)
    clust_features = features[sort_index]
    clust_pseu_labels = pseu_labels[sort_index]
    clust_id = id[sort_index]

    weighted_id = np.empty([0], dtype=int)
    weighted_pseu_label = np.empty([0], dtype=int)
    weights = np.empty([0])
    for i in range(num_class):
        class_feature = clust_features[clust_pseu_labels == i]
        class_id = clust_id[clust_pseu_labels == i]
        class_mean = np.mean(class_feature, axis=0)

        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-10)
        R=np.linalg.norm(class_feature,axis=1)[0]
        # class_dist=np.arccos(np.sum(class_feature / R * class_mean.reshape(-1, 256), axis=1))
        class_dist = 1 - np.sum(class_feature / R * class_mean.reshape(-1, 256), axis=1)
        class_dist = class_dist - np.min(class_dist)
        class_dist[2 * np.arange(len(class_dist) // 2)] = -1 * class_dist[2 * np.arange(len(class_dist) // 2)]
        weight = gauss_unif(class_dist.reshape(-1, 1))

        weights = np.hstack((weights, weight))
        weighted_id = np.hstack((weighted_id, class_id))
        weighted_pseu_label = np.hstack((weighted_pseu_label, np.ones_like(class_id, dtype=int) * i))

    return weighted_id,weighted_pseu_label,weights


def make_list(id,pseu_label,weights,list_path,save_path):
    lists = open(list_path).readlines()
    labeled_list = [lists[id[i]].split(' ')[0] + ' ' + str(pseu_label[i]) + ' '
                    + str(weights[i])  for i in range(len(id))]
    fw = open(save_path, 'w')
    for l in labeled_list:
        fw.write(l)
        fw.write('\n')

def make_weighted_pseudo_list(args,model):
    list_path = args.target_list
    if not os.path.exists('data/{}/pseudo_list'.format(args.dataset)):
        os.mkdir('data/{}/pseudo_list'.format(args.dataset))
    save_path = 'data/{}/pseudo_list/{}_{}_list.txt'.format(args.dataset,args.source,args.target)
    dsets = ImageList(open(list_path).readlines(), transform=image_test())
    dloader = DataLoader(dsets, batch_size=2*args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    features = torch.Tensor([])
    labels = torch.LongTensor([])
    pseu_labels = torch.LongTensor([])
    with torch.no_grad():
        for data in dloader:
            input = data[0]
            label = data[1]
            input = input.cuda()
            feature, outputs = model(input)

            features = torch.cat([features, feature.cpu()], dim=0)
            labels = torch.cat([labels, label], dim=0)
            pseu_labels = torch.cat([pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    weighted_id, weighted_pseu_label, weights= sample_weighting(features, labels, pseu_labels)
    make_list(weighted_id,weighted_pseu_label,weights, list_path, save_path)




