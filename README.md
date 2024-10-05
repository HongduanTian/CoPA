<center> 

# Mind the Gap Between Prototypes and Images in Cross-domain Finetuning

</center>
<center>

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]() [![Static Badge](https://img.shields.io/badge/Pub-NeurIPS'24-blue)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Static Badge](https://img.shields.io/badge/Slides%20-D76364)]() [![Static Badge](https://img.shields.io/badge/Poster%20-Ffa500)]() [![Static Badge](https://img.shields.io/badge/CN_Video%20-54b345)]() [![Static Badge](https://img.shields.io/badge/EN_Video%20-54b345)]()

</center>

This repository contains the source codes for reproducing the results of NeurIPS'24 paper:

[**Mind the Gap Between Prototypes and Images in Cross-domain Finetuning**]().

**Author List**: Hongduan Tian, Feng Liu, Zhanke Zhou, Tongliang Liu, Chengqi Zhang, Bo Han. 

## Introduction
<center>
<figure>
<img src=./illustrationfigures/pipeline.png/>
</center>
In cross-domain few-shot classification (CFC), recent works mainly focus on adapting a simple transformation head on top of a frozen pre-trained backbone with few labeled data to project embeddings into a task-specific metric space where classification can be performed by measuring similarities between image instance and prototype representations. Technically, an assumption implicitly adopted in such a framework is that the prototype and image instance embeddings share the same representation transformation. However, in this paper, we find that there naturally exists a gap, which resembles the modality gap, between the prototype and image instance embeddings extracted from the frozen pre-trained backbone, and simply applying the same transformation during the adaptation phase constrains exploring the optimal representations and shrinks the gap between prototype and image representations. To solve this problem, we propose a simple yet effective method, contrastive prototype-image adaptation (CoPA), to adapt different transformations respectively for prototypes and images similarly to CLIP by treating prototypes as text prompts. Extensive experiments on Meta-Dataset demonstrate that CoPA achieves the state-of-the-art performance more efficiently. Meanwhile, further analyses also indicate that CoPA can learn better representation clusters, enlarge the gap, and achieve minimal validation loss at the enlarged gap. 

<center>
<figure>
<img src=./illustrationfigures/gap.png/>
</center>




## Model Zoo
- [Single-domain networks (one for each dataset)](https://drive.google.com/file/d/1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9/view?usp=sharing)

- [A single universal network (URL) learned from 8 training datasets](https://drive.google.com/file/d/1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A/view?usp=sharing)


## Dependencies
This code requires the following software:
* Python 3.8
* PyTorch 1.7.1
* Torchvision 0.8.2
* TensorFlow 2.10
* tqdm 4.64.1
* tabulate 0.8.10

## Installation
* Clone or download this repository.
* Configure Meta-Dataset:
    * Follow the "User instructions" in the [Meta-Dataset repository](https://github.com/google-research/meta-dataset) for "Installation" and "Downloading and converting datasets".
    * To test unseen domain (out-of-domain) performance on additional datasets, i.e. MNIST, CIFAR-10 and CIFAR-100, follow the installation instruction in the [CNAPs repository](https://github.com/cambridge-mlg/cnaps) to get these datasets.


## Backbone Pre-training

### Train the Universal Representation Learning Network
1. The easiest way is to download [pre-trained URL model](https://drive.google.com/file/d/1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A/view?usp=sharing) provided by URL. To download the pretrained URL model, one can use `gdown` (installed by ```pip install gdown```) and execute the following command in the root directory of this project:
    ```
    gdown https://drive.google.com/uc?id=1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A && md5sum url.zip && unzip url.zip -d ./saved_results/ && rm url.zip
    
    ```

2. Alternatively, one can train the model from scratch: 1) train 8 single domain learning networks; 2) train the universal feature extractor as following. 

#### Train Single Domain Learning Networks
1. The easiest way is to download [pre-trained models](https://drive.google.com/file/d/1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9/view?usp=sharing) and use them to obtain a universal set of features directly. To download single domain learning networks, execute the following command in the root directory of this project:
    ```
    gdown https://drive.google.com/uc?id=1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9 && md5sum sdl.zip && unzip sdl.zip -d ./saved_results/ && rm sdl.zip
    ```

2. Alternatively, instead of using the pretrained models, one can train the models from scratch.
   To train 8 single domain learning networks, run:
    ```
    ./scripts/train_resnet18_sdl.sh
    ```


#### Train the Universal Feature Extractor
To learn the universal feature extractor by distilling the knowledge from pre-trained single domain learning networks, run: 
```
./scripts/train_resnet18_url.sh
```

## CoPA
### Meta-Testing with CoPA
This step would run CoPA procedure per task to learn the optimal task-specific parameters for each cross-domain few-shot classification task. Run:
```
./scripts/copa_pa.sh
```

### Meta-Testing with CoPA
This step would run CoPA procedure per task to learn the optimal task-specific parameters for each cross-domain few-shot classification task. Run:
```
./scripts/copa_tsa.sh
```
