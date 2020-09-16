sudo docker run --gpus '"device=0"' -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -p $1:$1 -v $(pwd):/workspace nvcr.io/nvidia/pytorch:20.06-py3 
