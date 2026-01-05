# On Membership Inference Attacks in Knowledge Distillation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview
This is the official repository for On Membership Inference Attacks in Knowledge Distillation.

## Installation
   ```bash
   pip install -r requirements.txt
   ```

## Running MIA
You can run MIA for a specific model using the following command:

```bash
cd src
python run_mia.py \
    --target_model <TARGET_MODEL> \
    --output_dir <OUTPUT_PATH> \
    --dataset <DATASET> \
    --sub_dataset <SUB_DATASET> \
    --num_shots <NUM_SHOTS>
```

For example, to run MIA on Pythia using ArXiv dataset, you can use:
```bash
python run_mia.py \
    --target_model "EleutherAI/pythia-160m" \
    --ref_model "EleutherAI/pythia-70m" \
    --output_dir ./out \
    --dataset "arxiv"
```

## Analysis
To analyze the testing results for one teacher-student pair, you can use the following command:

```bash
python analysis.py \
    --teacher_model_name <TEACHER_MODEL_NAME> \
    --teacher_model_testing_results_path <TEACHER_MODEL_TESTING_PATH> \
    --student_model_names <STUDENT_MODEL_NAMES> \
    --student_model_testing_results_paths <STUDENT_MODEL_TESTING_PATHS>
```

## Implementing Bottleneck and NoNorm
To distill student models with bottleneck and NoNorm implementations, you can use the following command:
```bash
cd src
python distill.py \
    --teacher-name <TEACHER> \
    --student-name <STUDENT> \
    --dataset <DATASET> \
    --save_path <SAVE_PATH> \
    --mode <MODE> \
    --bottleneck-dim <BOTTLENECK-DIM> \
    --batch-size <BATCH-SIZE> \
    --epochs <EPOCHS> \
    --lr <LEARNING_RATE>
```

`mode` can be chosen from "bottleneck", "nonorm", "none", and "all", where "all" is implementing both "bottleneck" and "nonorm".
