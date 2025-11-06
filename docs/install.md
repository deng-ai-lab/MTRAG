# Installation 🛠️
We recommend setting up a conda environment for the project:
```Shell
git clone https://github.com/deng-ai-lab/MTRAG.git
cd MTRAG
conda create -n mtrag python=3.10 -y
conda activate mtrag
pip install --upgrade pip 
pip install torch==2.1.2 torchvision==0.16.2 -i https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation--no-cache-dir
pip install -e .
```