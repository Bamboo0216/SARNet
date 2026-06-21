## Installation

Detailed environment configuration is provided in [environment.yml](environment.yml).

The commands below are provided as a practical reference. If you change the PyTorch or CUDA version, related packages may also need to be adjusted.

### Step-by-step installation

Start from the project root:

```bash
cd SGG_ToolKit
```

Create and activate the conda environment:

```bash
conda create -n SGG_ToolKit python=3.8
conda activate SGG_ToolKit
```

Install base dependencies:

```bash
conda install ipython scipy h5py
pip install ninja yacs cython matplotlib tqdm opencv-python overrides shapely ipdb
```

Install PyTorch. The example below uses CUDA 11.0:

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install `cocoapi`. Download `cocoapi.zip` from [STAR_OBJ_REL_WEIGHTS](https://huggingface.co/Zhuzi24/STAR_OBJ_REL_WEIGHTS/tree/main), unzip it under the project root, then run:

```bash
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../..
```

Install `apex`. Download `apex.zip` from [STAR_OBJ_REL_WEIGHTS](https://huggingface.co/Zhuzi24/STAR_OBJ_REL_WEIGHTS/tree/main), unzip it under the project root, then run:

```bash
cd apex
python setup.py install
cd ..
```

Install additional dependencies:

```bash
pip install torch-geometric==2.0.4 torch-scatter==2.0.7 torch-sparse==0.6.9
pip install numpy==1.23.5
```

Build the SGG-ToolKit package:

```bash
python setup.py build develop
```

Install the mmrotate / mmdetection components required by the STAR setting:

```bash
cd mmrote_RS
pip install openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.26.0
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```

### Important for HBB SGG

To run the HBB type of SGG, replace the `mmdet` package in the active conda environment with the project version:

- Source: `lib_mmdet/mmdet`
- Target: `<conda-env>/lib/python3.8/site-packages/mmdet`

For example, the target is usually similar to:

```text
/xxx/miniconda3/envs/SGG_ToolKit/lib/python3.8/site-packages/mmdet
```

### Common issues

1. If you see:

```text
AssertionError: MMCV==1.7.1 is used but incompatible. Please install mmcv>=1.4.5, <=1.6.0.
```

Install `mmcv==1.6.0` instead:

```bash
mim install mmcv==1.6.0
```

2. If you see:

```text
AssertionError: Egg-link /xxx/envs/SGG_A/lib/python3.8/site-packages/mmrotate.egg-link (to /xxx/mmrote_RS) does not match installed location of mmrotate (at /xxx/SGG_ToolKit/mmrote_RS)
```

Remove the stale `mmrotate.egg-info` directory, then reinstall `mmrote_RS` with `pip install -v -e .`.
