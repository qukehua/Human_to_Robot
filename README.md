
<h1 align="center">  </h1>





##  Environment
The project is developed under the following environment:
- Python 3.10 
- PyTorch 2.1
- CUDA 12.1




For installation of the project dependencies, please run:
```
conda create -n human_to_robot python=3.10
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
``` 


##  Training
You can train the model as follows:
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name HARPER_result.txt --layer-norm-axis spatial --with-normalization
```
where config files are located at `config/harper_config.yml`.
