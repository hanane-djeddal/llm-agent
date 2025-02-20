#!/bin/bash
#SBATCH --job-name=h100_testalce_att_paramkw
#SBATCH --output=jz_%j_%x.out
#SBATCH --error=errjz_%j_%x.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --account=fiz@h100

module purge
module load arch/h100
module load miniforge/24.9.0
export CONDA_PKGS_DIRS=$WORK/conda_envs
export CONDA_ENVS_PATH=/lustre/fswork/projects/rech/fiz/udo61qq/.conda/envs
conda activate llm-jz
export PYTHONPATH=$WORK/.local/lib/python3.10/site-packages:$PYTHONPATH
export HF_HOME=$WORK/.cache/huggingface
export JAVA_HOME=/lustre/fswork/projects/rech/fiz/udo61qq/libs/jdk-22.0.2
export PATH=$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
export JVM_PATH=/lustre/fswork/projects/rech/fiz/udo61qq/libs/jdk-22.0.2/lib/server/libjvm.so
export PYSERINI_CACHE=$WORK/.cache/pyserini

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 3 --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 4
