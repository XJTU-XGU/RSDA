# Robust Spherical Domain Adaptation
RSDA code for your own datasets other than office, office-home, and visda.
## Prerequisites:
Python==3.6.13 <br>
Pytorch ==1.5.1 <br>
Torchvision ==0.6.0 <br>
Numpy==1.19.2 <br>
Scipy==1.5.4 <br>
Scikit-learn==0.17.1 <br>

Before training, modify the root to you data at lines 46 and 55 in file "pre_process.py". 
## Initialization:
To obtain the initial model, run 
```
python train_init.py --source xxx --target xxx --num_class xxx --s_dset_path xxx --t_dset_path xxx
```
"--source": source domain name <br>
"--target": target domain name <br>
"--num_class": number of classes <br>
"--s_dset_path": path to the source data list <br>
"--t_dset_path": path to the target data list <br>
## Training:
Run
```
python train_with_pseudo_label.py --source xxx --target xxx --num_class xxx --s_dset_path xxx --t_dset_path xxx
```
## Citation:
```
@InProceedings{Gu_2020_CVPR,
author = {Gu, Xiang and Sun, Jian and Xu, Zongben},
title = {Spherical Space Domain Adaptation With Robust Pseudo-Label Loss},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
## Reference code:
Some parts of the code are borrowed from https://github.com/thuml/CDAN.
## Contact：
If you have any problem, please contact xianggu@stu.xjtu.edu.cn.
