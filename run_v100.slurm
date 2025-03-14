#!/bin/bash
#SBATCH --job-name=v100_alce_diversitysampling_contrastivesearch_queryonly
#SBATCH --output=jz_%j_%x.out    # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out  
#SBATCH --gres=gpu:8   #for v account max is 4  #SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00 # ### SBATCH --constraint=v100-32g
#SBATCH --cpus-per-task=20
#SBATCH --account=fiz@v100  ### SBATCH -C v100
#SBATCH --partition=gpu_p2l
module purge
module load arch/a100
module load pytorch-gpu/py3/2.0.1

export JAVA_HOME=/lustre/fswork/projects/rech/fiz/udo61qq/libs/jdk-22.0.2
export PATH=$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
export JVM_PATH=/lustre/fswork/projects/rech/fiz/udo61qq/libs/jdk-22.0.2/lib/server/libjvm.so
export PYSERINI_CACHE=$WORK/.cache/pyserini
export HF_HOME=$WORK/.cache/huggingface/
export PYSERINI_CACHE=$WORK/.cache/pyserini

cd /linkhome/rech/geniri01/udo61qq/Code/llm-agent/
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --nb_rounds 4 --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-param-knw-rag-agent-13b
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 4 --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 4  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-rag-agent-13b 
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 4  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-param-knw-rag-agent-13b
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 4  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama_testInference.py --nb_rounds 4 --inference_variant without_query  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b/ 

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 8  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --tag forced_rounds_retrival_with_answer_seg_13b_llama_corrected_v100 --retrieve_with_answer  --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testHagrid_forced_rounds_retrival_with_answer_seg_13b_llama_corrected_a100_8rounds_3docs.json
##CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama_testInference.py  --nb_rounds 8 --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --inference_variant empty_query

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --add_instruction --retrieve_with_answer --tag monot5  --ranker MonoT5
###--resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_with_params_high_diversity_sampling_using_answer_for_retrieval_8rounds_3docs.json 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_diversity_beam --gen_config '{"num_beams": 6, "num_beam_groups": 3, "diversity_penalty":1.2, "do_sample":false,"max_new_tokens": 1000}'
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_high_diversity_sampling  --gen_config '{"top_k":40,"top_p":0.9,"temperature":0.85, "do_sample":true, "max_new_tokens": 1000}' --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_with_params_high_diversity_sampling_using_answer_for_retrieval_8rounds_3docs.json 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_diversityhybrid  --gen_config '{"num_beams": 4, "top_k":50, "top_p":0.9, "temperature":0.8, "do_sample":true,  "max_new_tokens": 1000}'
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_contrastivesearch  --gen_config '{"penalty_alpha":0.6, "top_k":4, "max_new_tokens": 1000}'
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_diverse_sampling  --gen_config  '{"top_k":40,"top_p":0.9,"temperature":0.85, "do_sample":true, "max_new_tokens": 1000}' --diverse_query_only
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_contrastivesearch  --gen_config '{"penalty_alpha":0.6, "top_k":4, "max_new_tokens": 1000}'  --diverse_query_only
