## Learning Regional Purity for 3D Instance Segmentation on Point Clouds (ECCV2022)

Code for the paper **Learning Regional Purity for 3D Instance Segmentation on Point Clouds**, ECCV 2020.

**Authors**: Shichao Dong, Guosheng Lin, Tzu-yi Hung

## Introduction
3D instance segmentation is a fundamental task for scene understanding, with a variety of applications in robotics and AR/VR. Many proposal-free methods have been proposed recently for this task, with remarkable results and high efficiency. However, these methods heavily rely on instance centroid regression and do not explicitly detect object boundaries, thus may mistakenly group nearby objects into the same clusters in some scenarios. In this paper, we define a novel concept of "regional purity" as the percentage of neighboring points belonging to the same instance within a fixed-radius 3D space. Intuitively, it indicates the likelihood of a point belonging to the boundary area. To evaluate the feasibility of predicting regional purity, we design a strategy to build a random scene toy dataset based on existing training data. Besides, using toy data is a "free" way of data augmentation on learning regional purity, which eliminates the burdens of additional real data. We propose Regional Purity Guided Network (RPGN), which has separate branches for predicting semantic class, regional purity, offset, and size. Predicted regional purity information is utilized to guide our clustering algorithm. Experimental results demonstrate that using regional purity can simultaneously prevent under-segmentation and over-segmentation problems during clustering.

## Installation

### Requirements
* Python 3.8.0
* Pytorch 1.7.1
* CUDA 11.0

### Virtual Environment
```
conda create -n rpgn python==3.8
source activate rpgn
```

### Install `RPGN`

(1) Clone the RPGN repository.
```
git clone https://github.com/dsc1126/RPGN.git --recursive 
cd PPGN
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv). The repository is recursively downloaded at step (1). We use the version 1.2.1 of spconv. 

**Note:** We further modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use our modified `spconv`.

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```
* Run `cd dist` and use pip to install the generated `.whl` file.

(4) Compile the `rpgn_ops` library.
```
cd lib/rpgn_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.

## Acknowledgement
This repo is built upon several repos, e.g.,  [spconv](https://github.com/traveller59/spconv), [PointGroup](https://github.com/dvlab-research/PointGroup),  and [ScanNet](https://github.com/ScanNet/ScanNet). 

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (scdong@ntu.edu.sg).


