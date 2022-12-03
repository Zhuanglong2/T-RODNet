# T-RODNet: Transformer for Vehicular Millimeter-Wave Radar Object Detection


T-RODNet has been accepted by IEEE Transactions on Instrumentation and Measurement (TIM).  

![T-RODNet Overview](./assets/images/overview.jpg?raw=true)  

Please cite our paper if this repository is helpful for your research:  

Statement: This code is based on RODNet.[[Arxiv]](https://arxiv.org/abs/2102.05150)

## Installation

Create a conda environment for T-RODNet. Tested under Python 3.6, 3.7, 3.8.
```commandline
conda create -n trodnet python=3.* -y
conda activate trodnet
```

Install pytorch.  
**Note:** If you are using Temporal Deformable Convolution (TDC), we only tested under `pytorch<=1.4` and `CUDA=10.1`. 
Without TDC, you should be able to choose the latest versions. 
If you met some issues with environment, feel free to raise an issue.
```commandline
conda install pytorch=1.4 torchvision cudatoolkit=10.1 -c pytorch  # if using TDC
# OR
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch  # if not using TDC
```

Setup RODNet package.
```commandline
pip install -e .
```
**Note:** If you are not using TDC, you can rename script `setup_wo_tdc.py` as `setup.py`, and run the above command. 
This should allow you to use the latest cuda and pytorch version. 

## Evaluation  
If you want to install`cruw-devkit` package, please refer to [`cruw-devit`](https://github.com/yizhou-wang/cruw-devkit) repository for detailed instructions. 

**Note:** Our code have revised the 'cruw' evaluation package. You can use it in (evaluate/ex_evaluate_rod2021.py).    

## Prepare data for RODNet

Download [ROD2021 dataset](https://www.cruwdataset.org/download#h.mxc4upuvacso). 
Follow [this script](https://github.com/yizhou-wang/RODNet/blob/master/tools/prepare_dataset/reorganize_rod2021.sh) to reorganize files as below.  

**Note:** To facilitate testing and evaluation of network performance. We strongly recommend testing as follows:  

The **test set** uses the following 4 sequences from the training set:  
                     [2019_04_09_BMS1001, 2019_04_30_MLMS001, 2019_05_23_PM1S013, 2019_09_29_ONRD005].  
                     
The **train set** uses the remaining 36 sequences.  

```
data_root
  - sequences
  | - train---------------------------------------> 36 train set 
  | | - <SEQ_NAME>
  | | | - IMAGES_0
  | | | | - <FRAME_ID>.jpg
  | | | | - ***.jpg
  | | | - RADAR_RA_H
  | | |   - <FRAME_ID>_<CHIRP_ID>.npy
  | | |   - ***.npy
  | | - ***
  | | 
  | - test----------------------------------------> 4 test set 
  |   - <SEQ_NAME>
  |   | - RADAR_RA_H
  |   |   - <FRAME_ID>_<CHIRP_ID>.npy
  |   |   - ***.npy
  |   - ***
  | 
  - annotations
  | - train
  | | - <SEQ_NAME>.txt
  | | - ***.txt
  | - test
  |   - <SEQ_NAME>.txt
  |   - ***.txt
  - calib
```

Convert data and annotations to `.pkl` files.
```commandline
python tools/prepare_dataset/prepare_data.py \
        --config configs/<CONFIG_FILE> \
        --data_root <DATASET_ROOT> \
        --split train,test \
        --out_data_dir data/<DATA_FOLDER_NAME>
```

## Train models

```commandline
python tools/train.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --log_dir checkpoints/
```

## Inference
**Note:** Please change the 'out_path' in (test.py).  
```commandline
python tools/test.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --checkpoint <CHECKPOINT_PATH> \
        --res_dir results/
```

## Evaluate models  
In this section, we have created new evaluation tools to facilitate testing. Therefore there is no need to test by uploading the website.  
```commandline
python evaluate/ex_evaluate_rod2021.py
```