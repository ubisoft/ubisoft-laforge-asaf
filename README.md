# ubisoft-la-forge-ASAF

Pytorch implementation of [Adversarial Soft Advantage Fitting (ASAF)](https://proceedings.neurips.cc/paper/2020/hash/9161ab7a1b61012c4c303f10b4c16b2c-Abstract.html). See installation instructions and example commandlines below. If you find this code useful please consider citing the paper.

#### Bibtex:
```
@article{barde2020adversarial,
  title={Adversarial Soft Advantage Fitting: Imitation Learning without Policy Optimization},
  author={Barde, Paul and Roy, Julien and Jeon, Wonseok and Pineau, Joelle and Pal, Chris and Nowrouzezahrai, Derek},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Installation (Ubuntu):
* Create new python environment with version 3.7:
    ```
    conda create --name asaf_env python=3.7.4
    source activate asaf_env
    ```
  
  If on windows:
    * Install pytorch (CUDA 10.1):
    ```
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
 
    * Install pytorch (CPU Only):
    ```
    conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
    ```
      
* Install pip dependencies:
  ```
  cd ubisoft-la-forge-ASAF
  pip install -r requirements.txt
  mkdir storage
  ```

* Install `alfred`:
  ```
  mkdir -p $HOME/asaf_env/ext 
  cd $HOME/asaf_env/ext 
  git clone --depth 1 --branch v0.2 https://github.com/julienroyd/alfred 
  cd alfred
  pip install -e .
  ```

* Install `playground`:
  ```
  mkdir -p $HOME/asaf_env/ext 
  cd $HOME/asaf_env/ext/
  git clone https://github.com/PBarde/ASAF-playground
  cd ASAF-playground
  pip install -e .
  ```

* Install `mujoco`:
  * Create mujoco folder:
    ```
    mkdir $HOME/.mujoco
    ```
  
  * Download mujoco 1.5 binaries:
    ```
    cd $HOME/.mujoco
    wget https://www.roboti.us/download/mjpro150_linux.zip
    unzip mjpro150_linux.zip
    rm mjpro150_linux.zip
    ```
  
  * Copy-paste mujoco lisence key:
    ```
    cd $HOME/.mujoco
    touch mjkey.txt
    vim mjkey.txt
    ```
    
  * Add these environment variables to `.bashrc` (don't forget to `source .bashrc` afterwards):
    ```
    export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
    export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mjpro150/
    export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin
    ```

  * Install `mujoco-py`:
    ```
    mkdir -p $HOME/asaf_env/ext 
    cd $HOME/asaf_env/ext 
    git clone https://github.com/openai/mujoco-py
    cd mujoco-py
    git checkout 9ea9bb000d6b8551b99f9aa440862e0c7f7b4191
    pip install -e .
    ```
    
* Test installation:
  ```
  python -c "import alfred"
  python -c "import pommerman"
  python -c "import mujoco_py"
  ```

## Expert data
* Download the archive from Google Drive
  ```
  cd ubisoft-la-forge-ASAF
  pip install gdown
  gdown https://drive.google.com/uc?id=1Zj686J3Dfc0lydXsccnYv9wN9ucG4zsb
  unzip data.zip -d data
  touch data/__init__.py
  ```

## BUG
Sometimes it's not running due to `mujocopylock`. You can solve simply by removing lockfile. See [this line](https://github.com/openai/mujoco-py/issues/424#issuecomment-534907857).

## Running trainings
There are two training scripts `irl/train.py` and `direct_rl/train.py`. The script `main.py` will automatically call the appropriate trainer based on the 
`alg_name` that you provide. You can find the list of available algorithms and environments in `alg_task_lists.py`.

We give here some examples on how to launch runs from the paper. All the hyper-parameters come from the appendix section of the paper.
### Continuous control (MuJoCo)
**ASAF-1 on hopper**
```
python main.py --alg_name asaf-1X --task_name hopper-c --demos_name expert_demo_25.pkl --max_transitions 2000000 --transitions_between_saves 5000 --d_transitions_between_updates 2000 --d_batch_size 100 --d_grad_norm_clip 1. --d_epochs_per_update 30 --d_lr 0.001
```
**ASAF-w on walker2d**
```
python main.py --alg_name asaf-wX --task_name walker2d-c --demos_name expert_demo_25.pkl --max_transitions 2000000 --transitions_between_saves 5000 --d_transitions_between_updates 2000 --d_batch_size 20 --d_grad_norm_clip 1. --d_epochs_per_update 10 --d_lr 0.001 --window_size 100 --window_stride 1
```
**ASAF on halfcheetah**
```
python main.py --alg_name asaf-fullX --task_name halfcheetah-c --demos_name expert_demo_25.pkl --max_transitions 2000000  --transitions_between_saves 5000 --d_episodes_between_updates 25 --d_batch_size 10 --d_grad_norm_clip 10. --d_epochs_per_update 50 --d_lr 0.001
```
**SQIL on ant**
```
python main.py --alg_name sqil-c --task_name ant-c --demos_name expert_demo_25.pkl --max_transitions 2000000  --transitions_between_saves 5000 --transitions_between_updates 1 --batch_size 256 --lr 0.0003
```
**GAIL + PPO on hopper**
```
python main.py --alg_name gailXppo --task_name hopper-c --demos_name expert_demo_25.pkl --max_transitions 2000000 --transitions_between_saves 5000 --d_transitions_between_updates 2000 --d_batch_size 2000 --d_grad_norm_clip 1. --d_epochs_per_update 5 --gradient_penalty_coef 1. --d_lr 0.011 --lr 0.000018 --batch_size 200 --grad_norm_clip 1. --lamda 0.98 --transitions_between_updates 2000 --epochs_per_update 5 --update_clip_param 0.2 --critic_lr_coef 0.25
```

### Discrete control (Pommerman)
**ASAF-1**
```
python main.py --alg_name asaf-1X --task_name learnablevsRandomPacifist1v1empty --demos_name expertDemo150_winsFrom0_nDifferentInit1.pkl --max_episodes 50000 --episodes_between_saves 500 --d_episodes_between_updates 10 --d_batch_size 256 --d_epochs_per_update 10 --d_lr 0.0001 
```
**ASAF-w**
```
python main.py --alg_name asaf-wX --task_name learnablevsRandomPacifist1v1empty --demos_name expertDemo150_winsFrom0_nDifferentInit1.pkl --max_episodes 50000 --episodes_between_saves 500 --d_episodes_between_updates 10 --d_batch_size 5 --d_epochs_per_update 10 --d_lr 0.0002 --window_size 32 --window_stride 32
```
**ASAF**
```
python main.py --alg_name asaf-fullX --task_name learnablevsRandomPacifist1v1empty --demos_name expertDemo150_winsFrom0_nDifferentInit1.pkl --max_episodes 50000 --episodes_between_saves 500 --d_episodes_between_updates 10 --d_batch_size 5 --d_epochs_per_update 10 --d_lr 0.0007
```
**SQIL**
```
python main.py --alg_name sqil --task_name learnablevsRandomPacifist1v1empty --demos_name expertDemo150_winsFrom0_nDifferentInit1.pkl --max_episodes 50000 --episodes_between_saves 500 --transitions_between_updates 10 --batch_size 256 --lr 0.00019 --replay_buffer_length 100000 --warmup 1280 --grad_norm_clip 0.2 --tau 0.05
```
**GAIL + PPO**
```
python main.py --alg_name gailXppo --task_name learnablevsRandomPacifist1v1empty --demos_name expertDemo150_winsFrom0_nDifferentInit1.pkl --max_episodes 50000 --episodes_between_saves 500 --d_episodes_between_updates 10 --d_batch_size 256 --d_epochs_per_update 10 --d_lr 9.3e-7 --lr 0.00015 --batch_size 256 --lamda 0.95 --episodes_between_updates 10 --epochs_per_update 10 --update_clip_param 0.2 --critic_lr_coef 0.5
```
**AIRL + PPO**
```
python main.py --alg_name airlXppo --task_name learnablevsRandomPacifist1v1empty --demos_name expertDemo150_winsFrom0_nDifferentInit1.pkl --max_episodes 50000 --episodes_between_saves 500 --d_episodes_between_updates 10 --d_batch_size 256 --d_epochs_per_update 10 --d_lr 3.1e-7 --lr 0.00017 --batch_size 256 --lamda 0.95 --episodes_between_updates 10 --epochs_per_update 10 --update_clip_param 0.2 --critic_lr_coef 0.5
```

You will find the runs under `/storage/` and you can use `evaluate.py` to do rollouts with the learned policy and make gifs. 
