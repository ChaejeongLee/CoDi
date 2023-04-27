# CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis
This code is the official implementation of "CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis".
(https://arxiv.org/abs/2304.12654)

## Requirements
Run the following to install requirements:
```setup
conda env create --file environment.yaml
```

## Usage
* Train and evaluate CoDi through `main.py`:
```sh
main.py:
  --data: tabular dataset
  --eval : train or eval
  --logdir: Working directory
```

## Training
* You can train our CoDi from scratch by run:
```bash
python main.py --data heart --logdir CoDi_exp
```

## Evaluation
* By run the following script, you can reproduce our experimental result: 
    binary classification result of CoDi on Heart in Table 10. 
```bash
python main.py --data heart --eval True --logdir exp_heart
```
