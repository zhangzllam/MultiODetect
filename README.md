# MultiODetect
Single-Image-Based Detection of Rotating Multi-Opening Objects
Download the GitHub repository and its dependencies:

Installation:

WORK_DIR=/path/to/work/directory/
cd $WORK_DIR
git clone <your-github-repo-url>.git
PROJECT_DIR=$WORK_DIR/MultiODetect

cd $PROJECT_DIR

conda create -y -n <your-env-name> python=3.8
conda activate <your-env-name>
conda install -y setuptools==69.5.1 mkl=2024.0 pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
