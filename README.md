If you find our code is useful, please cite our paper

# Usage
- For all kinds of models in the model_zoo: 
    - access model.nn_mass will return the NN_Mass of the model 
## MLP
### NN_Mass vs. Test Accuracy
- Evaluate the test accuracy of MLP with self-defined topology/architecture
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
  | --tc            |     the number of tc |
  | --dataset       | the type of dataset |
  | --make_dataset   |      generate/regenerate the synthetic dataset or not |
  | --train_log_file |   the name of file used to record the training/test record of MLPs |
  | --res_log_file  | the name of file used to record the training/test record of MLPs |
  | --iter_times  | the number of iteration times to train the same architecture |


- Example: train a 8-layer MLP with 8 neurons and tc=10 per layer on MNIST dataset

    * python train_mlp.py --depth=8 --width=8 --tc=10 --dataset='MNIST' 


### NN_Mass vs. LDI
- Calculate the LDI (mean singular value of Jacobians) of MLP with self-defined topology/architecture. Currently we only support on MNIST dataset. 
- The usage is similar to train_mlp.py
    - python ldi.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
|   -h, --help          |  show this help message and exit
|   --batch_size BATCH_SIZE    |      Number of samples per mini-batch
|   --epochs EPOCHS     |      Number of epoch to train
|   --depth DEPTH        |     the depth (number of FC layers) of the MLP
|   --width WIDTH         |    the width (number of neurons per layers) of the MLP
|   --num_seg NUM_SEG     |    the number of segmentation for the synthetic dataset
|   --tc TC               |    the number of tc
|   --dataset DATASET     |    the type of dataset
|   --sigma_log_file SIGMA_LOG_FILE    |    the name of file used to record the LDI record of MLPs
|   --iter_times ITER_TIMES     |   the number of iteration times to calculate the LDI of  the same architecture

- Example: Calculate a 8-layer MLP with 8 neurons and tc=10 per layer on MNIST dataset

    * python ldi.py --depth=8 --width=8 --tc=10 --dataset='MNIST' 

## CNN
### CIFAR-10/100
- Evaluate the test accuracy of MLP with self-defined topology/architecture
- usage: 
    - python train_cifar.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
  | -h, --help      |      show this help message and exit |
  | --batch_size    | Number of samples per mini-batch |
  | --epochs        | Number of epoch to train |
  | --lr            |     Learning rate |
  | --depth         |  the depth (number of FC layers) of the MLP |
  | --width         |  the width (number of neurons per layers) of the MLP |
  | --num_seg       | the number of segmentation for the synthetic dataset (currently we support 'linear' and 'circle' dataset)|
  | --tc            |     the number of tc |
  | --dataset       | the type of dataset |
  | --make_dataset   |      generate/regenerate the synthetic dataset or not |
  | --train_log_file |   the name of file used to record the training/test record of MLPs |
  | --res_log_file  | the name of file used to record the training/test record of MLPs |
  | --iter_times  | the number of iteration times to train the same architecture |


- Example: train a 8-layer MLP with 8 neurons and tc=10 per layer on MNIST dataset

    * python train_mlp.py --depth=8 --width=8 --tc=10 --dataset='MNIST' 

### ImageNet


