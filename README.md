# PDANet : Pyramid Dual Attention Network for Semantic Segmentation

By Guoqiang Liu, lgqfhwy@zju.edu.cn.


This code is an implementation of the experiments on ADE20K in the PDANet. We implement our method 
based on open source [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)




## Environment
The code is developed under the following configurations.
- Hardware: 1-8 GPUs (with at least 12G GPU memories) (change ```[--gpus GPUS]``` accordingly)
- Software: Ubuntu 16.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0***

## Quick start: 
1. Here is a simple demo to do inference on a single image:
```bash
chmod +x demo_test.sh
./demo_test.sh
```
This script downloads a trained model (ResNet50dilated + PPM_deepsup) and a test image, runs the test script, and saves predicted segmentation (.png) to the working directory.



2. Train a model (default: ResNet18dilated + PDAM). During training, checkpoints will be saved in folder ```ckpt```.
```bash
python3 train.py --gpus GPUS
```

- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

For example:

* Train resnet18dilated + PDAM (Pyramid Dual Attention Module)
```bash
python3 train.py --gpus GPUS \
--arch_decoder pdam \
```





## Evaluation
1. Evaluate a trained model on the validation set. ```--id``` is the folder name under ```ckpt``` directory. ```--suffix``` defines which checkpoint to use, for example ```_epoch_20.pth```. Add ```--visualize``` option to output visualizations as shown in teaser.
```bash
python3 eval_multipro.py --gpus GPUS --id MODEL_ID --suffix SUFFIX
```

For example:


* Evaluate ResNet18dilated + PDAM
```bash
python3 eval_multipro.py --gpus GPUS \
    --id MODEL_ID --suffix SUFFIX --arch_encoder resnet18dilated --arch_decoder pdam \
```



    
