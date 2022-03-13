# Robust Spherical Domain Adaptation
Code for paper "Xiang Gu, Jian Sun, Zongben Xu, **Spherical Space Domain Adaptation with Robust Pseudo-label Loss**, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020".
## Prerequisites:
Python==3.6.13 <br>
Pytorch ==1.5.1 <br>
Torchvision ==0.6.0 <br>
Numpy==1.19.2 <br>
Scipy==1.5.4 <br>
Scikit-learn==0.17.1 <br>
## Initialization:
To obtain the initial model, run 
```
python train_init.py --source train --target validation
```
The returned accuracy is the result of method "DANN+S" in our paper. The trained initial model can also be found at https://drive.google.com/drive/folders/1leBh0cjjKyIXx5crrBs_r8_M-kfogmmy?usp=sharing.
## Training RSDA-DANN:
Download the initial model and put it into the folder "./save/init_model". Then run
```
python train_with_pseudo_label.py --source train --target validation
```
## Training RSDA-DANN with class-aware sampling:
Download the initial model and put it into the folder "./save/init_model". Then run
```
python train_with_pseudo_label_cas.py --source train --target validation
```
## Results
We run the code on a single NVIDIA Tesla V100 GPU, the results (using ReaNet50 backbone) are as follows.<br>
| |Synthetic-Real|
|----|:----:|
|RSDA-DANN|81.9|
|RSDA-DANN-CAS|82.9|
## Citation:
```
@InProceedings{Gu_2020_CVPR,
author = {Gu, Xiang and Sun, Jian and Xu, Zongben},
title = {Spherical Space Domain Adaptation With Robust Pseudo-Label Loss},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

@ARTICLE{9733209,
  author={Gu, Xiang and Sun, Jian and Xu, Zongben},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Unsupervised and Semi-supervised Robust Spherical Space Domain Adaptation}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2022.3158637}
}
```
## Reference code:
Some parts of the code are borrowed from https://github.com/thuml/CDAN.
## Contact：
If you have any problem, please contact xianggu@stu.xjtu.edu.cn.
