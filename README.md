# Robust Spherical Domain Adaptation
Code for paper "Xiang Gu, Jian Sun, Zongben Xu, **Spherical Space Domain Adaptation with Robust Pseudo-label Loss**, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020".
## Prerequisites:
Python3 <br>
Pytorch ==1.0.0 <br>
torchvision ==0.2.3 <br>
Numpy <br>
Scipy <br>
argparse <br>
## Dataset:
Offie-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
Please modify the path of the images in every ".txt" in "./data/office".
## Training:
Please run main.py as following according to your own gpu numbers.<br>
```
python3 main.py --source amazon --target dslr --source_list ./data/office/amazon_list.txt --target_list ./data/office/dslr_list.txt --gpu_id 0 
```
## Citation:
To be added.
## Reference code:
Some parts of the code are built based on https://github.com/thuml/CDAN.
## Contact：
If you have any problem, please contact xiangu@stu.xjtu.edu.cn.
