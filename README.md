# ProteoBayes
Official implementation of [Rationalized All-Atom Protein Design with Unified Multi-modal Bayesian Flow](https://openreview.net/forum?id=3p4272zl7q).

Table of Content
- [Environment Set Up](#environment-set-up)
- [Benchmark Metrics Reimplementation](#benchmark-metrics-reimplementation)
- [Training](#training)
## Environment Set Up
Run the following script to install most of the packages.
```
conda env create -f probayes_env.yml
```
Next we need to manually install several pip packages. 
### Install PyRosetta
The PyRosetta version we use is `pyrosetta-2024.35+release.45abd6a-cp310-cp310-linux_x86_64.whl`.
You need to download this [file](https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux.wheel/pyrosetta-2024.35+release.45abd6a-cp310-cp310-linux_x86_64.whl) into your server and install it:
```
pip install pyrosetta-2024.35+release.45abd6a-cp310-cp310-linux_x86_64.whl
```
### Install pdbfix
```
pip install git+https://github.com/pandegroup/pdbfixer.git
```
### Install torch-scatter
```
pip install torch-scatter==2.1.2+pt20cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
### Install the project
```
pip install -e .
```
### Download Datasets
Download the pre-processed dataset files in this [link](https://drive.google.com/file/d/18_zHYOZYiVIMKx0sXe_2xYNFa7l4r3Tc/view?usp=sharing).
And unzip it in the project root.
### Download Pre-Compute Cache
Download the pre-computed cache files in this [link](https://drive.google.com/file/d/18_zHYOZYiVIMKx0sXe_2xYNFa7l4r3Tc/view?usp=sharing).
And unzip it in the project root.

### Download Checkpoint
We provide our checkpoints and the designed PDB files for benchmark (PepBench, PepBDB, RAbD) evaluation [here](https://drive.google.com/drive/folders/1rnqwKgjFAtKvpQXL-fZxp0T10OF5RvxK?usp=drive_link).


### Project Structure
After installation, the project structure should be like:
```
/probayes
|-- README.md
|-- cache_files
|-- ckpts
|-- configs
|-- logs
|-- openfold
|-- probayes
|-- probayes.egg-info
|-- probayes_data
|-- probayes_data.zip
|-- probayes_env.yml
|-- remote
|-- scripts
|-- setup.py
|-- train.py
|-- train_antibody.py
|-- train_antibody_ddp.py
|-- train_pep.py
|-- train_pep_ddp.py
`-- wandb
```

## Benchmark Metrics Reimplementation
You may need to add the execuation permission for DockQ evaluation.
```
chmod +x /GenSIvePFS/wuhl/probayes/remote/PepGLAD/evaluation/DockQ/fnat
chmod +x /GenSIvePFS/wuhl/probayes/remote/ppflow/bin/TMscore/TMscore
```
All training and evaluation scripts can be found in `scripts/`. For reimplementing the benchmark metric scores:
1. Peptide codesign
```
source scripts/eval_ckpt_peptide.sh
```
2. Peptide Binding Conformation Generation / Folding
```
source scripts/eval_ckpt_folding.sh
```
3. Antibody design
```
source scripts/eval_ckpt_antibody.sh
```
You can choose the desired dataset by switching the `CKPT_DIR` variable in the bash file.
## Training
We provide our training scripts here:
1. Peptide codesign
```
source scripts/train_ddp_antibody_codesign.sh -d pepbench
```
2. Peptide Binding Conformation Generation / Folding
```
source scripts/train_ddp_pep_folding.sh -d pepbench
```
3. Antibody design
```
source scripts/train_ddp_antibody_codesign.sh
```
The default setting requires 4x80GB GPUs for 10~24 hours. You can check the benchmark scores in wandb.

# Acknowledgements

We would like to express our gratitude to the following repositories for their valuable contributions:

*   [**diffab**](https://github.com/luost26/diffab)
*   [**dyMEAN**](https://github.com/THUNLP-MT/dyMEAN)
*   [**PepGLAD**](https://github.com/THUNLP-MT/PepGLAD)
*   [**ppflow**](https://github.com/EDAPINENUT/ppflow)
*   [**PepFlow**](https://github.com/Ced3-han/PepFlowww)