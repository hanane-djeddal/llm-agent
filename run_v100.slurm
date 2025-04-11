#!/bin/bash
#SBATCH --job-name=v100p2_hagrid_instructp2###statp2 ###alce_instrcutp2##adding_userquery ##hagrid_trainedwithinstruct#hagrid_wodecomp  ##alce_trainedwithintrcut ##forcingrounds_stat ###param ###withoutqInference_param ###  withoutqInference_param monot5_instruct###diversity_sampling_continue ###contrastivesearch_p4 ### diversity_sampling_querycontinue
#SBATCH --output=jz_%j_%x.out    # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out  
#SBATCH --gres=gpu:8   #for v account max is 4  #SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00 # ### SBATCH --constraint=v100-32g
#SBATCH --cpus-per-task=20
#SBATCH --account=fiz@v100  ### SBATCH -C v100
#SBATCH --partition=gpu_p2
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

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama_testInference.py --nb_rounds 4 --inference_variant empty_query  --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-param-knw-rag-agent-13b ##llama-2-chat-hagrid-att-rag-agent-13b

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 8  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --tag forced_rounds_retrival_with_answer_seg_13b_llama_corrected_v100 --retrieve_with_answer  --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testHagrid_forced_rounds_retrival_with_answer_seg_13b_llama_corrected_a100_8rounds_3docs.json
##CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama_testInference.py  --nb_rounds 8 --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --inference_variant empty_query

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --add_instruction --retrieve_with_answer --tag monot5  --ranker MonoT5 --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_monot5_instruction_prompt_using_answer_for_retrieval_8rounds_3docs.json
###--resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_monot5_instruction_prompt_using_answer_for_retrieval_8rounds_3docs.json 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_diversity_beam --gen_config '{"num_beams": 6, "num_beam_groups": 3, "diversity_penalty":1.2, "do_sample":false,"max_new_tokens": 1000}'
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_high_diversity_sampling  --gen_config '{"top_k":40,"top_p":0.9,"temperature":0.85, "do_sample":true, "max_new_tokens": 1000}' --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_with_params_high_diversity_sampling_using_answer_for_retrieval_8rounds_3docs.json 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_diversityhybrid  --gen_config '{"num_beams": 4, "top_k":50, "top_p":0.9, "temperature":0.8, "do_sample":true,  "max_new_tokens": 1000}' --resume_fom_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_with_params_diversityhybrid_using_answer_for_retrieval_8rounds_3docs.json
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_contrastivesearch  --gen_config '{"penalty_alpha":0.6, "top_k":4, "max_new_tokens": 1000}' --startindex 841 --stopindex 948  ### --startindex 280 --stopindex 560   561-840  841-948
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_diverse_sampling  --gen_config  '{"top_k":40,"top_p":0.9,"temperature":0.85, "do_sample":true, "max_new_tokens": 1000}' --diverse_query_only --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_with_params_diverse_sampling_using_answer_for_retrieval_diverse_query_only_8rounds_3docs.json
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --retrieve_with_answer --tag with_params_contrastivesearch  --gen_config '{"penalty_alpha":0.6, "top_k":4, "max_new_tokens": 1000}'  --diverse_query_only


### forcing round param & stat
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8  --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-rag-agent-13b --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_llama-2-chat-hagrid-att-rag-agent-13b_8rounds_3docs.json ##llama-2-chat-hagrid-att-rag-agent-13b  ##llama-2-chat-hagrid-att-param-knw-rag-agent-13b  
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 8  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-rag-agent-13b  --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testHagrid_llama-2-chat-hagrid-att-rag-agent-13b_8rounds_3docs.json ##intr_testHagrid_llama-2-chat-hagrid-att-param-knw-rag-agent-13b_8rounds_3docs.json ##llama-2-chat-hagrid-att-rag-agent-13b


#### trained with instruct
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-segnli-instruct-rag-agent-13b  --add_instruction --retrieve_with_answer --resume_from_file  /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testasqa_llama-2-chat-hagrid-att-segnli-instruct-rag-agent-13b_instruction_prompt_using_answer_for_retrieval_8rounds_3docs.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama.py --nb_rounds 8  --ragnroll_model_name /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-segnli-instruct-rag-agent-13b  --add_instruction --retrieve_with_answer  --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testHagrid_llama-2-chat-hagrid-att-segnli-instruct-rag-agent-13b_instruction_prompt_using_answer_for_retrieval__8rounds_3docs.json

####hagrid empty query
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_hagrid_llama_testInference.py  --nb_rounds 8 --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b  --inference_variant empty_query --resume_from_file /lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/intr_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsempty_query.json 

## appending userquery to subqueries
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python test_alce_normal_llama.py --nb_rounds 8 --query_file  /lustre/fswork/projects/rech/fiz/udo61qq/ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  /lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-seg-proba-rag-agent-13b --add_user_query
