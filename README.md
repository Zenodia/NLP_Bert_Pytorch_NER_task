# Fine-tune Bert to Swedish for NER task and make prediction 
this repo containing example guide on 
1. how to use the existing huggingface BERT to do Ner 
2. how to train ( fine-tune) a BERT model with custom dataset (swedish NER task)

## Requirements -
For the execution of all examples, we need a multi-GPU server (such as an NVIDIA DGX). 

This repository use docker container from  NGC repo (go to https://ngc.nvidia.com/ ) and run and tested on RTX-8000 GPU with Ubuntu 18.04 as OS. 

You will need at least the following minimum setup:
- Supported Hardware: NVIDIA GPU with Pascal Architecture or later (Pascal, Volta, Turing)
- Any supported OS for nvidia docker
- [NVIDIA-Docker 2](https://github.com/NVIDIA/nvidia-docker)
- NVIDIA Driver ver. 450.66 (However, if you are running any Tesla Graphics, you may use driver version 396, 384.111+, 410, 418.xx or 440.30)


### NVIDIA Docker used
- pytorch 20.06-py3 NGC container 
For a full list of supported systems and requirements on the NVIDIA docker, consult [this page](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/#framework-matrix-2020).

#### Step 0 -  pull the git repo 
```bash 
git clone  https://github.com/Zenodia/pytorch_DALI.git
cd into the pytorch_DALI folder
```

#### Step 1 - run docker image pulled from NGC repo
```bash
sudo docker run --gpus '"device=0"' -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -p $1:$1 -v $(pwd):/workspace nvcr.io/nvidia/pytorch:20.06-py3 
 
```


#### Step 2 - build the environment from within the docker image ran in Step 1 
follow these 3 substeps to set up the environment 
##### Step 2.1 
`pip install torch torchvision tqdm ipywidgets jupyterlab & jupyter nbextension enable --py widgetsnbextension`
##### Step 2.2 
`pip3 install -r requirements.txt`
##### Step 2.3 install huggingface transfomers
following this guide [transformer_install](https://huggingface.co/transformers/master/installation.html)
`git clone https://github.com/huggingface/transformers.git & cd transformers & pip install -e .`

#### Step 3 - Launch the Jupyter Notebook
```bash
bash 2_launch_jupyter.sh <port_number>
```
To run jupyter notebook in the host browser , remember to use the same port number you specify in docker run on step 1


#### Step 4 - call out a preferred browser and use jupyter as UI to interact with the running docker
call out firefox ( or other default browser )
type in in the browser url: `https://0.0.0.0:<port_number>`
If you are using a remote server, change the url accordingly: `http://you.server.ip.address:<port_number>`
!
#### use huggingface default models to do NER prediction task

```3_Swedish_NER_HuggingFace.ipynb```


#### find-tune Bert model with NER task and make prediction 
cd into folder NER_train_predict
and run through the notebook 
```Swedish_NER_train_from_scratch_with_predict.ipynb```
you should be able to see the prediction on a Swedish sentence as below screenshot shown

![BERT_NER_predict](<./pics/NER_predict.JPG>) 


### References 
https://huggingface.co/transformers/master/installation.html
https://github.com/kamalkraj/BERT-NER 
https://huggingface.co/blog/how-to-train
