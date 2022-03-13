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
python train_init.py --source Art --target Clipart
```
The returned accuracy is the result of method "MSTN+S" in our paper. The trained initial model can also be found at https://drive.google.com/drive/folders/1Rt6crgL4-n9ZGzXLjGw0wmvk4GxlUpGI?usp=sharing.
## Training:
Download the initial model and put it into the folder "./save/init_model". Then run
```
python train_with_pseudo_label.py --source Art --target Clipart
```
## Results
We run the code on a single NVIDIA Tesla V100 GPU, the results are as follows. The results are higher than those reported in the original paper.<br>
|Method |Ar-Cl|Ar-Pr|Ar-Rw|Cl-Ar|Cl-Pr|Cl-Rw|Pr-Ar|Pr-Cl|Pr-Rw|Rw-Ar|Rw-Cl|Rw-Pr|Avg|
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|MSTN+S   |57.7|73.7|79.1|65.2|74.8|74.4|65.0|58.4|81.0|73.4|59.7|83.4|70.5|
|RSDA-MSTN|59.9|78.6|81.5|68.5|77.7|77.9|67.5|60.9|82.3|75.3|61.5|85.7|73.1|<br>
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
