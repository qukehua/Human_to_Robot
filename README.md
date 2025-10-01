
<h1 align="center">  </h1>





##  Environment
The project is developed under the following environment:
- Python 3.8
- PyTorch 2.4
- CUDA 12.4

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 


##  Training
You can train the model as follows:
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name HARPER_result.txt --layer-norm-axis spatial --with-normalization
```
where config files are located at `configs/harper_config.py`.
