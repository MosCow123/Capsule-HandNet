# Capsule-HandNet

## Installation

Install h5py for Python:
```bash
  sudo apt-get install libhdf5-dev
  sudo pip install h5py
```
--
Install Chamfer distance package:
```bash
  cd models/nndistance
  python build.py install
``` 

## Usage
### Data download
Download the MSRA Hand Gesture database at:
https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0

Unzip it and put folders "P0"~"P9" in the "data/cvpr15_MSRAHandGestureDB" directory.

Please cite "Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang and Jian Sun. Cascaded Hand Pose Regression. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015", if you use this database.

### Data preprocess
    cd preprocess
    matlab preprocess.m 
Please cite this paper if you use this code.
"Liuhao Ge, Yujun Cai, Junwu Weng and Junsong Yuan. Hand PointNet: 3D Hand Pose Estimation using Point Sets. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2018."


### Train and evaluation

Pretrained Model trained on "P1"~"P9": [BaiduNetDesk Link](https://pan.baidu.com/s/1AhDSa_G39tcEssLhABff1g)

put encoder model in 'AutoEncoder/results/P0' and regression model in 'results/P0'.
#### AutoEncoder Training and evaluating
    cd AutoEncoder
    python train.py
    python eval.py
set "capsule model" as the saved model.

#### Hand pose regression
    python train.py
    python eval.py
    
## References
Our capsule model is based on <a href="https://github.com/yongheng1991/3D-point-capsule-networks" target="_blank">3D-point-capsule-networks</a>..
    



