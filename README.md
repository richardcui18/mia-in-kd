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
python run.py \
    --target_model <TARGET_MODEL> \
    --output_dir <OUTPUT_PATH> \
    --dataset <DATASET>
```

For example, to run MIA on Pythia using ArXiv dataset, you can use:
```bash
python run.py \
    --target_model "EleutherAI/pythia-160m" \
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
