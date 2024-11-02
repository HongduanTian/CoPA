<h1 align='center'> Mind the Gap Between Prototypes and Images in Cross-domain Finetuning</h1>

<p align='center'>
<a href="http://arxiv.org/abs/2410.12474"><img src="https://img.shields.io/badge/arXiv-2410.12474-b31b1b.svg" alt="Paper"></a> <a href="https://neurips.cc/"><img src="https://img.shields.io/badge/Pub-NeurIPS'24-blue" alt="Conf"></a> <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Liscence"></a> <a href="https://hongduantian.github.io/_pages/data/slides/NeurIPS24_CoPA.pdf"><img src="https://img.shields.io/badge/Slides%20-D76364" alt="Slides"></a> <a href=""><img src="https://img.shields.io/badge/Poster%20-Ffa500" alt="Poster"></a> <a href=""><img src="https://img.shields.io/badge/CN_Video%20-54b345" alt="CN_Video"></a> <a href=""><img src="https://img.shields.io/badge/EN_Video%20-54b345" alt="EN_Video"></a>

</p>

This repository contains the source codes for reproducing the results of NeurIPS'24 paper: [**Mind the Gap Between Prototypes and Images in Cross-domain Finetuning**](http://arxiv.org/abs/2410.12474).

**Author List**: Hongduan Tian, Feng Liu, Zhanke Zhou, Tongliang Liu, Chengqi Zhang, Bo Han. 

## Introduction

<p align='center'>
<img src=./illustrationfigures/pipeline.png width=600/>
</p>

In cross-domain few-shot classification (CFC), recent works mainly focus on adapting a simple transformation head on top of a frozen pre-trained backbone with few labeled data to project embeddings into a task-specific metric space where classification can be performed by measuring similarities between image instance and prototype representations. Technically, an assumption implicitly adopted in such a framework is that the prototype and image instance embeddings share the same representation transformation. However, in this paper, we find that there naturally exists a gap, which resembles the modality gap, between the prototype and image instance embeddings extracted from the frozen pre-trained backbone, and simply applying the same transformation during the adaptation phase constrains exploring the optimal representations and shrinks the gap between prototype and image representations. To solve this problem, we propose a simple yet effective method, contrastive prototype-image adaptation (CoPA), to adapt different transformations respectively for prototypes and images similarly to CLIP by treating prototypes as text prompts. Extensive experiments on Meta-Dataset demonstrate that CoPA achieves the state-of-the-art performance more efficiently. Meanwhile, further analyses also indicate that CoPA can learn better representation clusters, enlarge the gap, and achieve minimal validation loss at the enlarged gap. 

<center>
<figure>
<img src=./illustrationfigures/gap.png/>
</center>

## Dependencies
In our experiments, the main dependences required are the following libraries:
```
Python 3.6 or greater (Ours: Python 3.8)
PyTorch 1.0 or greater (Ours: torch=1.7.1, torchvision=0.8.2)
TensorFlow 1.14 or greater (Ours: TensorFlow=2.10)
tqdm (Ours: 4.64.1)
tabulate (0.8.10)
```

## Dataset
- Follow [Meta-Dataset repository](https://github.com/google-research/meta-dataset) to prepare `ILSVRC_2012`, `Omniglot`, `Aircraft`, `CU_Birds`, `Textures (DTD)`, `Quick Draw`, `Fungi`, `VGG_Flower`, `Traffic_Sign` and `MSCOCO` datasets.

- Follow [CNAPs repository](https://github.com/cambridge-mlg/cnaps) to prepare `MNIST`, `CIFAR-10` and `CIFAR-100` datasets.



## Backbone Pretraining
In this paper, we follow [URL](https://arxiv.org/pdf/2103.13841.pdf) and use ResNet-18 as the frozen backbone in all our experiments. For reproduction, two ways are provided:

__Train your own backbone.__ You can train the ResNet-18 backbone from scratch by yourself. The pretraining mainly contains two phases: domain-specific pretraining and universal backbone distillation.

To train the single domain-specific learning backbones (on 8 seen domains), run:
```
./scripts/train_resnet18_sdl.sh
```

Then, distill the model by running:
```
./scripts/train_resnet18_url.sh
```

__Use the released backbones.__ URL repository has released both universal backbone and single domain backbone. For simplicity, you can directly use the released model.
- [Single-domain networks (one for each dataset)](https://drive.google.com/file/d/1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9/view?usp=sharing)

- [A single universal network (URL) learned from 8 training datasets](https://drive.google.com/file/d/1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A/view?usp=sharing)

The backbones can be downloaded with the above links. To download the pretrained URL model, one can use `gdown` (installed by ```pip install gdown```) and execute the following command in the root directory of this project:
```
gdown https://drive.google.com/uc?id=1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9 && md5sum sdl.zip && unzip sdl.zip -d ./saved_results/ && rm sdl.zip  # Universal backbone
gdown https://drive.google.com/uc?id=1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A && md5sum url.zip && unzip url.zip -d ./saved_results/ && rm url.zip  # Domain specific backbones
```
In this way, the backbones are donwnloaded. Please create the ```./saved_results``` directory and place the backbone weights in it. 

## Evaluate CoPA
To evaluate the CoPA, you can run:
```
./scripts/copa_pa.sh
```
Specifically, the running command is:
```
python copa_pa.py --model.name=url \
                  --model.dir ./url \
                  --test.type=standard \
                  --encoder.type=linear \
                  --SCE.tau=2.0 \
                  --seed=42 \
                  --exp_dir_name=linear_all \
                  --experiment.name=seed42
```
The hyperparameters can be modified for different experiments:
- `model_name: ['sdl', 'url']`: `sdl` means using single domain backbone; `url` means using universal backbone.
- `model.dir`: Path to the backbone weights.
- `test.type ['standard', '5shot', '1shot']`: Different task modes. `standard` means vary-way vary-shot tasks; `5shot` means vary-way 5-shot tasks; `1shot` means 5-way 1-shot tasks.
- `encoder.type ['linear', 'vit']`: Select different transformation modules to run CoPA.
- `SCE.tau`: The temperature coefficient used in SCE loss.
- `seed`: The random seed. All our results are the average of seed 41-45.

To evaluate the __CoPA+TSA__, you can run:
```
./scripts/copa_tsa.sh
```
Specifically, the running command is:
```
python copa_tsa.py --model.name=url \
                  --model.dir ./url \
                  --test.type=standard \
                  --encoder.type=linear \
                  --SCE.tau=2.0 \
                  --seed=42 \
                  --exp_dir_name=linear_all \
                  --experiment.name=seed42
```
### Evaluate Pre-classifier Alignment (PA)
To evaluate Pre-classifier Alignment (PA), which is the typical case of URL, run:

```
./scripts/test_resnet18_pa.sh
```

To evaluate URL with task-specific adapters (TSA), which is an modified case of URL, run:

```
./scripts/test_resnet18_tsa.sh
```

## Acknowledgement
 
 The repository is built mainly upon these repositories:
 
- [VICO-UoE/URL [1]](https://github.com/VICO-UoE/URL);
- [google-research/meta-dataset [2]](https://github.com/google-research/meta-dataset)

[1] Li et al. [Universal representation learning from multiple domains for few-shot classification](https://arxiv.org/pdf/2103.13841), ICCV 2021.

[2] Triantafillou et al. [Meta-dataset: A dataset of datasets for learning to learn from few examples](https://arxiv.org/pdf/1903.03096), ICLR 2020.

## Citation
```
@inproceedings{tian2024mind,
    title={Mind the gap between prototypes and images in cross-domain finetuning},
    author={Hongduan Tian and Feng Liu and Zhanke Zhou and Tongliang Liu and Chengqi Zhang and Bo Han},
    booktitle={Advances of Neural Information Processing Systems (NeurIPS)},
    year={2024}
}
```
