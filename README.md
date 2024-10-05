<p align="center">

  <h1 align="center">Towards Global Localization using Multi-Modal Object-Instance
Re-Identification</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=fiFj4AwAAAAJ" target="_blank">Aneesh Chavan</a><sup>*1</sup>,</span> 
    <a href="SECOND AUTHOR PERSONAL LINK" target="_blank">Vaibhav Agrawal</a><sup>*1</sup>,</span>
    <a href="https://scholar.google.com/citations?user=vsqwwPYAAAAJ&hl=en" target="_blank">Vineeth Bhat</a><sup>†1</sup>,<br>
    <a href="https://scholar.google.com/citations?user=La1bvRsAAAAJ&hl=en" target="_blank">Sarthak Chittawar</a><sup>†1</sup>,
    <a href="THIRD AUTHOR PERSONAL LINK" target="_blank">Siddharth Srivastava</a><sup>3</sup>,
    <a href="https://scholar.google.com/citations?user=Q8cTLNMAAAAJ&hl=en" target="_blank">Chetan Arora</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=QDuPGHwAAAAJ&hl=en" target="_blank">K Madhava Krishna</a><sup>1</sup>
    <br>
    <sup>1</sup>Robotics Research Centre, IIIT Hyderabad, 
    <sup>2</sup>IIT Delhi, 
    <sup>3</sup>Typeface Inc.<br>
    <sup>*</sup>equal contribution, <sup>†</sup>equal contribution
  </p>
  <h2 align="center">Submitted to ICRA 2025</h2>
  <h3 align="center"><a href="https://github.com/instance-based-loc/instance-based-loc">Code</a> | <a href="https://arxiv.org/abs/2409.12002">Paper</a> | <a href="https://github.com/instance-based-loc/instance-based-loc/blob/main/datasets.md">Datasets</a> | <a href="https://instance-based-loc-machine.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://instance-based-loc-machine.github.io/static/images/pipeline-new.drawio-cropped-compressed.png" alt="Logo" width="100%">
  </a>
</p>
<!-- <p align="center">
<strong>OpenMask3D</strong> is a zero-shot approach for 3D instance segmentation with open-vocabulary queries.
Guided by predicted class-agnostic 3D instance masks, our model aggregates per-mask features via multi-view fusion of CLIP-based image embeddings.
</p> -->
<br>

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
