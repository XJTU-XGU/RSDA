import argparse
from solvers import train_init,train
from gaussian_uniform.weighted_pseudo_list import make_weighted_pseudo_list
import copy
import torch
import os

def main(args):
    args.log_file.write('\n\n###########  initialization ############')
    acc, model_temp = train_init(args)
    best_acc = acc
    best_model = copy.deepcopy(model_temp)
    for stage in range(args.stages):
        print('\n\n########### stage : {:d}th ##############\n\n'.format(stage))
        args.log_file.write('\n\n########### stage : {:d}th    ##############'.format(stage))
        make_weighted_pseudo_list(args, model_temp)
        acc,model_temp = train(args)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model_temp)
    torch.save(best_model,'snapshot/save/final_best_model.pk')
    print('final_best_acc:{:.4f}'.format(best_acc))
    return best_acc,best_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset',type=str,default='office')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target',type=str,default='dslr')
    parser.add_argument('--source_list', type=str, default='data/office/amazon_list.txt', help="The source dataset path list")
    parser.add_argument('--target_list', type=str, default='data/office/dslr_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--num_class',type=int,default=31,help='the number of classes')
    parser.add_argument('--stages',type=int,default=6,help='the number of alternative iteration stages')
    parser.add_argument('--max_iter',type=int,default=5000)
    parser.add_argument('--gamma',type=float,default=0.1,help='coefficient of entropy')
    parser.add_argument('--batch_size',type=int,default=36)
    parser.add_argument('--log_file')
    args = parser.parse_args()
    if args.source == 'amazon':
        args.gamma = 1.0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not os.path.exists('snapshot'):
        os.mkdir('snapshot')
    if not os.path.exists('snapshot/{}'.format(args.output_dir)):
        os.mkdir('snapshot/{}'.format(args.output_dir))
    log_file = open('snapshot/{}/log.txt'.format(args.output_dir),'w')
    log_file.write('dataset:{}\tsource:{}\ttarget:{}\n\n'
                   ''.format(args.dataset,args.source,args.target))
    args.log_file = log_file

    main(args)




