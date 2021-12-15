import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import torch.nn.functional as F
from pre_process import ImageList,ImageListPseudo,ClassAwareSamplingSTDataset
import copy
import numpy as np
import random
import torch.optim as optim
import pseudo_labeling
import tqdm



def mae_loss(output,label,weight,q=1.0):
    one_hot_label=torch.zeros(output.size()).scatter_(1,label.cpu().view(-1,1),1).cuda()
    mask=torch.eq(one_hot_label,1.0)
    output=F.softmax(output,dim=1)
    mae=(1-torch.masked_select(output,mask)**q)/q
    # print(q,mae)
    return torch.sum(weight*mae)/(torch.sum(weight)+1e-10)

def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in tqdm.trange(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    # _, predict = torch.max(all_output, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    class_num = all_output.shape[1]
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    subclasses_correct = np.zeros(class_num)
    subclasses_tick = np.zeros(class_num)
    correct = 0
    tick = 0
    for i in range(predict.size()[0]):
        subclasses_tick[int(all_label[i])] += 1
        if predict[i].float() == all_label[i]:
            correct += 1
            subclasses_correct[predict[i]] += 1
    accuracy = correct * 1.0 / float(all_label.size()[0])
    subclasses_result = np.divide(subclasses_correct, subclasses_tick)
    print("========accuracy per class==========")
    print(subclasses_result, subclasses_result.mean())

    return accuracy,subclasses_result.mean(),predict


def train(config):
    source = config["source"]
    target = config["target"]
    ## set pre-process
    prep_dict = {}
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"],root=config["root"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"],root=config["root"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["target_pseudo"] = ImageListPseudo(open('new_list/' + source + '_' + target + '_list.txt').readlines(), \
                                transform=prep_dict["target"],root=config["root"])
    dset_loaders["target_pseudo"] = DataLoader(dsets["target_pseudo"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["classaware"] = ClassAwareSamplingSTDataset(open(data_config["source"]["list_path"]).readlines(),
                                                      open('new_list/' + source + '_' + target + '_list.txt').readlines(),
                                                      transform=prep_dict["target"],
                                                      num_class=config["network"]["params"]["class_num"],
                                                      num_sample_per_class=3,
                                                      dataset_length=200,
                                                      root=config["root"])
    dset_loaders["classaware"] = DataLoader(dsets["classaware"], batch_size=config["network"]["params"]["class_num"], num_workers=4)
    dsets["target_pseudo"] = ImageListPseudo(open('new_list/' + source + '_' + target + '_list.txt').readlines(), \
                                transform=prep_dict["target"],root=config["root"])
    dset_loaders["target_pseudo"] = DataLoader(dsets["target_pseudo"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                              transform=prep_dict["test"],root=config["root"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ad_net = network.AdversarialNetworkSp(base_network.output_num(), 1024,second=True,radius=config["network"]["params"]["radius"])
    ad_net = ad_net.cuda()

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net)
        base_network = nn.DataParallel(base_network)

    parameter_classifier=[base_network.get_parameters()[2]]
    parameter_feature=base_network.get_parameters()[0:2]+ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer_classfier = optimizer_config["type"](parameter_classifier, \
                                         **(optimizer_config["optim_params"]))
    optimizer_feature = optimizer_config["type"](parameter_feature, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer_feature.param_groups:
        param_lr.append(param_group["lr"])
    param_lr.append(optimizer_classfier.param_groups[0]["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    best_acc = 0.0
    best_acc_class = 0.0
    best_model = copy.deepcopy(base_network)

    for i in range(config["iterations"]):
        if i % config[ "test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc,acc_class,predict = image_classification_test(dset_loaders, base_network)
            temp_model = base_network
            if acc_class > best_acc_class:
                best_acc_class = acc_class
                best_model = copy.deepcopy(temp_model)
            if temp_acc > best_acc:
                best_acc = temp_acc
                # best_model = copy.deepcopy(temp_model)
            log_str = "iter: {:05d},\ttemp_acc:{:.4f},\t best_acc: {:.4f},\t acc_class:{:.4f}, \t best_acc_class:{:.4f}" \
                          "".format(i,temp_acc,best_acc,acc_class,best_acc_class)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
            dsets["classaware"].update_selected_classes(predict)
        if (i + 1) % config["snapshot_interval"] == 0:
            if not os.path.exists("save/rsda_model"):
                os.makedirs("save/rsda_model")
            torch.save(best_model, 'save/rsda_model/'+source+'_'+target+'.pkl')

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer_classfier = lr_scheduler(optimizer_classfier, i, **schedule_param)
        optimizer_feature = lr_scheduler(optimizer_feature, i, **schedule_param)

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        if i % len(dset_loaders["target_pseudo"]) == 0:
            iter_target_pseudo = iter(dset_loaders["target_pseudo"])
        if i%len(dset_loaders["classaware"]) == 0:
            iter_classaware = iter(dset_loaders["classaware"])

        inputs_source, labels_source = iter_source.next()
        inputs_target,_ = iter_target.next()
        pinputs_target, plabels_target,gammas,sigmas = iter_target_pseudo.next()
        adv_source,adv_target = iter_classaware.next()
        adv_source = adv_source.view(-1,3,224,224)
        adv_target = adv_target.view(-1,3, 224, 224)

        inputs_source, inputs_target, pinputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(),\
                                                                      pinputs_target.cuda(),labels_source.cuda()
        gammas,sigmas=gammas.type(torch.Tensor).cuda(),sigmas.type(torch.Tensor).cuda()
        adv_source, adv_target = adv_source.cuda(),adv_target.cuda()

        weight_c = gammas
        weight_c[weight_c < 0.5] = 0.0

        features_source, outputs_source = base_network(inputs_source)
        features_target, _ = base_network(inputs_target)
        adv_inputs = torch.cat((adv_source, adv_target), dim=0)
        features,_ = base_network(adv_inputs)
        transfer_loss = loss.DANN(features, ad_net)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        _, outputs_target = base_network(pinputs_target)
        classifier_loss_target = mae_loss(outputs_target,plabels_target,weight_c)

        fc = copy.deepcopy(base_network.fc)
        for param in fc.parameters():
            param.requires_grad = False
        softmax_tar_out = torch.nn.Softmax(dim=1)(fc(features_target))
        H = torch.mean(loss.Entropy(softmax_tar_out))

        lam=network.calc_coeff(i,max_iter=2000)
        total_loss = classifier_loss+classifier_loss_target+transfer_loss+config["tradeoff_ent"]*H

        optimizer_classfier.zero_grad()
        optimizer_feature.zero_grad()
        total_loss.backward()
        optimizer_classfier.step()
        optimizer_feature.step()

        print('step:{: d},\t,class_loss:{:.4f},\t,transfer_loss:{:.4f},'
              '\t,class_loss_t:{:.4f}, \t H:{:.4f}'.format(i, classifier_loss.item(),
                                              transfer_loss.item(),classifier_loss_target.item(),H.item()))
    return best_acc,best_acc_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for RSDA-DANN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=str, default='train', choices=["train"])
    parser.add_argument('--target', type=str, default='validation', choices=["validation"])
    parser.add_argument('--test_interval', type=int, default=200, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--stages', type=int, default=10, help="training stages")
    parser.add_argument('--radius', type=float, default=8.5, help="radius")
    args = parser.parse_args()
    root = "/public/home/guxiang/guxiang/datasets"
    # root = "/data/guxiang/dataset"
    s_dset_path = '{}/visda-2017/'.format(root) + args.source + '.txt'
    t_dset_path = '{}/visda-2017/'.format(root) + args.target + '.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = {}
    config["source"] = args.source
    config["target"] = args.target
    config["gpu"] = args.gpu_id
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/rsda"
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"],args.source+"_"+args.target+ "_log_cas.txt"), "a")

    config["prep"] = {'params':{"resize_size":256, "crop_size":224}}
    config["network"] = {"name":network.ResNetCos, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True,"class_num":12,"radius":args.radius} }
    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }
    config["data"] = {"source":{"list_path":s_dset_path, "batch_size":64}, \
                      "target":{"list_path":t_dset_path, "batch_size":64}, \
                      "test":{"list_path":t_dset_path, "batch_size":128}}
    config["out_file"].flush()
    config["iterations"] = 10001
    config["tradeoff_ent"] = 1.0
    config["root"]=root

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    best_acc = 0.0
    best_acc_class = 0
    for k in range(0,args.stages,1):
        if k==0:
            config["iterations"] = 801
        else:
            config["iterations"] = 5001
        print('time:',k)
        config["out_file"].write('\n---time:'+str(k)+'---\n')
        pseudo_labeling.make_new_list(args.source, args.target, iter_times=k,root=config["root"])
        temp_acc,temp_acc_class=train(config)
        if best_acc<temp_acc:
            best_acc=temp_acc
        if best_acc_class<temp_acc_class:
            best_acc_class=temp_acc_class
    print("best_acc:",best_acc)
    config["out_file"].write('\nbest_acc:{:.4f}\tbest_acc_class:{:.4f}'.format(best_acc,best_acc_class))
