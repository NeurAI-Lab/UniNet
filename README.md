# UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks

This is the official code for the ICCV'21 DeepMTL workshop Paper ["UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks"](https://arxiv.org/abs/2009.08325) by 
[Naresh Kumar Gurulingan](https://scholar.google.com/citations?user=6XoXurUAAAAJ&hl=en), [Elahe Arani](https://www.researchgate.net/profile/Elahe-Arani) and [Bahram Zonooz](https://scholar.google.com/citations?hl=en&user=FZmIlY8AAAAJ)


## Abstract

Scene understanding is crucial for autonomous systems which intend to operate in
the real world. Single task vision networks extract information only based on
some aspects of the scene. In multi-task learning (MTL), on the other hand, 
these single tasks are jointly learned, thereby providing an opportunity for 
tasks to share information and obtain a more comprehensive understanding. 
To this end, we develop UniNet, a unified scene understanding network that 
accurately and efficiently infers vital vision tasks including object detection,
semantic segmentation, instance segmentation, monocular depth estimation, and 
monocular instance depth prediction. As these tasks look at different semantic
and geometric information, they can either complement or conflict with each
other. Therefore, understanding inter-task relationships can provide useful cues
to enable complementary information sharing. We evaluate the task relationships
in UniNet through the lens of adversarial attacks based on the notion that they
can exploit learned biases and task interactions in the neural network. 
Extensive experiments on the Cityscapes dataset, using untargeted and targeted
attacks reveal that semantic tasks strongly interact amongst themselves, and the
same holds for geometric tasks. Additionally, we show that the relationship
between semantic and geometric tasks is asymmetric and their interaction becomes
weaker as we move towards higher-level representations.


![alt text](images/Encoder_Decoder.pdf "UniNet Encoder-Decoder")

For details, please see the
[Paper](https://arxiv.org/abs/2108.04584) 


## Environment:

conda_env.yml file can be used to create an anaconda environment to run the code.


Train models:

To train the UniNet model on the Cityscapes dataset: <br />

python train.py --batch-size 8 --workers 8 --data-folder /data/input/datasets/Cityscapes --crop-size 512 1024 --checkname test_cs --output-dir /volumes2/naresh.gurulingan/uninet/ --dataset uninet_cs --pretrained --config-file ./resources/uninet_5tasks.yaml <br />

Pretrained checkpoints for DLA34 can be found at: https://github.com/aim-uofa/AdelaiDet/blob/master/configs/FCOS-Detection/README.md#fcos-real-time-models 
The path to the model can be added in ./resources/uninet_5tasks.yaml in the "PRETRAINED_PATH" field.

Evaluate models:

Models can be evaluated using --eval-only arg along with train script.



### Cite Our Work
If you find the code useful in your research, please consider citing our paper:

<pre>
@InProceedings{Gurulingan_2021_ICCV, <br />
    author    = {Gurulingan, Naresh Kumar and Arani, Elahe and Zonooz, Bahram}, <br />
    title     = {UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships Through the Lens of Adversarial Attacks}, <br />
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops}, <br />
    month     = {October}, <br />
    year      = {2021}, <br />
    pages     = {2239-2248} <br />
}
</pre>

### License
This project is licensed under the terms of the MIT license.
