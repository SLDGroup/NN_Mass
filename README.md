Offical code for 'How does topology influence gradient propagation and model performance of deep networks with DenseNet-type skip connections?' (CVPR 2021) [ArXiv link](https://arxiv.org/pdf/1910.00780.pdf)

# Usage
- For all kinds of models in the model_zoo: 
    - access ```model.nn_mass``` will return the NN_Mass value of the model 
## MLP
### NN_Mass vs. Test Accuracy
- Evaluate the test accuracy of MLPs with random concatenation-type skip connections
- usage: 
    - python train_mlp.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
  | -h, --help      |      show this help message and exit |
  | --batch_size    | Number of samples per mini-batch |
  | --epochs        | Number of epoch to train |
  | --lr            |     Learning rate |
  | --depth         |  the depth (number of FC layers) of the MLP |
  | --width         |  the width (number of neurons per layers) of the MLP |
  | --num_seg       | the number of segmentation for the synthetic dataset (currently we support 'linear' and 'circle' dataset)|
  | --tc            |     maximum number of candidate channels/neurons from previous layers that can supply concatenation-type skip connections |
  | --dataset       | the type of dataset |
  | --make_dataset   |      generate/regenerate the synthetic dataset or not |
  | --train_log_file |   the name of file used to record the training/test record of MLPs |
  | --res_log_file  | the name of file used to record the training/test record of MLPs |
  | --iter_times  | the number of iteration times to train the same architecture |


- Example: train a 8-layer MLP with 8 neurons per layer and tc=10 on MNIST dataset

    * python train_mlp.py --depth=8 --width=8 --tc=10 --dataset='MNIST' 


### NN_Mass vs. LDI
- Calculate the LDI (mean singular value of Jacobians) of MLPs with random concatenation-type skip connections. 
- Currenlty, our code only could calculate the LDI of MLP on MNIST dataset. 
- The usage is similar to train_mlp.py
    - python ldi.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
|   -h,           |  show this help message and exit
|   --batch_size     |      Number of samples per mini-batch
|   --epochs      |      Number of epoch to train
|   --depth         |     the depth (number of FC layers) of the MLP
|   --width          |    the width (number of neurons per layers) of the MLP
|   --num_seg      |    the number of segmentation for the synthetic dataset
|   --tc                |    maximum number of candidate channels/neurons from previous layers that can supply concatenation-type skip connections
|   --dataset DAASET     |    the type of dataset
|   --sigma_log_file     |    the name of file used to record the LDI record of MLPs
|   --iter_times      |   the number of iteration times to calculate the LDI of  the same architecture

- Example: Calculate the LDI of a 8-layer MLP with 8 neurons per layer and tc=10 on MNIST dataset

    * python ldi.py --depth=8 --width=8 --tc=10 --dataset='MNIST' 

## CNN
### CIFAR-10/100
- Train and evaluate the test accuracy of MLP with self-defined topology/architecture
- Currently we support DensetNet-type CNNs ('regular_dense') and DensetNet-type +Depthwisw Conv CNNs. ('dense_depth')
- usage: 
    - python train_cifar.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
  | -h, --help      |      show this help message and exit |
  |--arch          |  network architecture |
  |--wm              |  width multipler of CNN cells|
  |--num_cells  |                        number of cells|
  | --cell_depth  |                        number of layers for each cell|
  |--tc1        |    within the first cell, maximum number of candidate channels/neurons from previous layers that can supply concatenation-type skip connections |
  |--tc2        |    within the second cell,   maximum number of candidate channels/neurons from previous layers that can supply concatenation-type skip connections |
  |--tc3         |   within the third cell,   maximum number of candidate channels/neurons from previous layers that can supply concatenation-type skip connections |
  |--tc4        |    within the fourth cell,   maximum number of candidate channels/neurons from previous layers that can supply concatenation-type skip connections |
|--dataset  |    dataset|
- Example: train a 3-cell wm=2 8-layer per cell DensetNet-type CNN with tc[10,20,30] for each cell on CIFAR10 dataset

    * python train_cifar.py --num_cells=3 --cell_depth=8 --wm=2 --tc1=10 --tc2=20 --tc3=40 --dataset='cifar10' --arch='regular_dense'

### ImageNet
We reuse some code from [mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv2.pytorch)\
Currently, we support the following networks (use ```python imagenet.py -a network_name``` to select the networks):
```
mobilenet_v2
resnet18
resnet34
resnet50
resnet101
resnet152
resnext50_32x4d
resnext101_32x8d
wide_resnet50_2
wide_resnet101_2
```

#### Example: Train MobileNet-v2
```
python imagenet.py \
    -a mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --epochs 150 \
    --lr-decay cos \
    --lr 0.05 \
    --wd 4e-5 \
    -c <path-to-save-checkpoints> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -j <num-workers>
```

####  Example: Test MobileNet-v2
```shell
python imagenet.py \
    -a mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --weight <pretrained-pth-file> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -e
```
## Dependency
```Please check the environment.sh ```\
Note: the installation of pytorch depends on your OS version and GPU types.
\
\
\
\
if you find our code useful, please consider citing our paper:
```
@article{bhardwaj2019does,
  title={How Does Topology of Neural Architectures Impact Gradient Propagation and Model Performance?},
  author={Bhardwaj, Kartikeya and Li, Guihong and Marculescu, Radu},
  journal={arXiv preprint arXiv:1910.00780},
  year={2019}
}
```

