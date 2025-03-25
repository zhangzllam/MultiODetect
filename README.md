# MultiODetect
Single-Image-Based Detection of Rotating Multi-Opening Objects


## Installation:

Download the GitHub repository and its dependencies:

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
install pytorch_wavelets 
`git clone https://github.com/fbcotter/pytorch_wavelets cd pytorch_wavelets pip install .`

## Run Example
 After installing, you can download a dataset (Rot-Multi-opening): https://doi.org/10.5281/zenodo.15080626
 
 Once you downloaded data fragment, you can test the code.
 Run the following command:
 ```
python main.py --batch_size 128 --epochs 200 --lr 0.001  --res_dir .\Result --data_root .\rotating_no_tur_filter_test --gamma 0.87
 ```

 
