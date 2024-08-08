# instance-based-loc

## Coding practices to follow

_Remove in the final code release_

1. Add an issue for every task that you'll be integrating
2. Create an appropriately named branch (or `dev-$USER` if no specific task) for all work
3. Create a pull request, link the issue that you're solving and then merge WHEN you're done adding the things mentioned in the issue.

Please do NOT push to main - we shouldn't have to rebase/other stuff for every few commits.

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
