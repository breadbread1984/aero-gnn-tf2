# Introduction

this project is to predict attribute of molecule with AERO GNN

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## Create dataset

```shell
python3 create_dataset.py --input_csv <path/to/csv> --output_dir dataset [--head 1] [--channels 64]
```

## Train AERO GNN

```shell
python3 train.py --dataset dataset
```

