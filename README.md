# PAMod

Welcome to the official repository of PAMod: "[PAMod: Modeling Cyclical Shifts via Phase-Amplitude Modulation for Non-stationary Time Series Forecasting]", which has been accepted to NeurIPS 2026. 


## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n PAMod python=3.8
conda activate PAMod
pip install -r requirements.txt
```

### Data Preparation
 
All datasets are publicly available. See the Appendix of this submission for access details.
Create a separate folder named ```./dataset/data_name``` and place all the CSV/NPZ files in this directory.
**Note**: Place the CSV/NPZ files directly into this directory, such as "./dataset/ETT-samll/ETTm1.csv".

### Training Scripts

The training scripts for replicating the PAMod results are available at:

```
./scripts/PAMod
```

### Quick Reproduction the Main Results

You can reproduce all the main results of PAMod with the following code snippet.

```
sh run_main.sh
```

For your convenience, we have provided the **execution results** of "sh run_main.sh":
```
./result.txt
```
