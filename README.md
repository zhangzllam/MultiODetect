# MultiODetect
Single-Image-Based Detection of Rotating Multi-Opening Objects
Download the GitHub repository and its dependencies:

## Installation:

```bash

WORK_DIR=/path/to/work/directory/
cd $WORK_DIR
git clone git@github.com:zhangzllam/MultiODetect.git
PROJECT_DIR=$WORK_DIR/MultiODetect

cd $PROJECT_DIR

conda create -y -n MultiODetect python=3.7
conda activate MultiODetect
conda install -y pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cudatoolkit=11.6 -c pytorch

```


