[readme.md](https://github.com/user-attachments/files/27563092/readme.md)
# ProjRes: Code Repository for "Toward Efficient Membership Inference Attacks against Federated Large Language Models: A Projection Residual Approach"

This repository contains the official implementation of the experiments and evaluations described in our paper **"Toward Efficient Membership Inference Attacks against Federated Large Language Models: A Projection Residual Approach"**.

## 📁 Repository Structure

The code is organized to mirror the structure of the paper. Specifically:

- Each subsection in the **Experimental Evaluation** section corresponds to a dedicated folder.
- Inside each folder:
  - The main Python scripts (`.py` files) contain the **experiment code** used to run the corresponding evaluation.
  - A `results/` subdirectory stores the **raw results and outputs** reported in the paper.
  - The script `results/plot_figure.py` generates the figures presented in the paper from the stored results.

```text
section_4.2/ # Reproduces Figure 4 in the paper
├── MIA_Bert_On_CoLA.py
├── MIA_Bert_On_IMDB.py
├── ...
├── results/
│   ├── mia_bert_cola_results_16.json
│   ├── mia_bert_IMDB_results_16.json
│   ├── ...
│   └── plot_figure.py  
├── auxiliary_function/
│   ├── Bert_insert_Adapter.py
│   └── ...
└── ...

section_4.3/ # Reproduces Table 6 in the paper
├── MIA_Qwen_Adapter1.2.py
├── ...
├── results/
│   ├── mia_Qwen_Adapter1.2_results.json
│   ├── ...
│   └── plot_figure.py  
├── auxiliary_function/
│   ├── Bert_insert_Adapter.py
│   └── ...
└── ...

...

```


## 🧪 Reproducibility

All experiments are fully reproducible:

1. Use the provided environment (see below) to ensure compatibility.
2. Run the experiment scripts in each section folder to regenerate results.
3. Execute `results/plot_figure.py` in any `results/` directory to reproduce the corresponding figure from the paper.


## ⚙️ Environment Setup

All code was developed and tested under the Python environment specified in [`environment.txt`](environment.txt).

To recreate the environment using Conda:

```bash
conda create --name MIA_env --file environment.txt
conda activate MIA_env
```





