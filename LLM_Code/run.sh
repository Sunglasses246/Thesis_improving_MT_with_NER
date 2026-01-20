#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1 
#SBATCH --time=00:30:00

#SBATCH -o out.o
#SBATCH -e err.e
${XCT_PATH} COT_code/Llama/Llama_3.1_8B_COT_IT.py