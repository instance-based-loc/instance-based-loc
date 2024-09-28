# instance-based-loc

## Code Release

Will be released by 00:00:00 AOT, 1st October, 2024.

## Setup

### Clone all submodules

```bash
git submodule update --init
```

### Setup conda environment

```bash
conda env create -f environment.yml
conda activate dator
```

### Setup additional modules

Please clone the repo recursively to clone all the submodules as well.

#### RAM Model

```bash
cd object_memory/recognize-anything
pip install -e .
```

#### Grounding Dino and SAM

```bash
cd object_memory/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.8 # export CUDA_HOME=/path/to/cuda-11.3/ for others
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
```

_NOTE_: Update the environment YAML before merging any PR. Remove the `prefix` property from the YAML file as well.

### Download weights

```bash
bash bash_scripts/download_ram_sam_weights.sh 
```

## Overall Documentation

Will be added
