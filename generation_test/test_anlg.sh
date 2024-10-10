#!/bin/bash

# tasks=("classification" "generation")
tasks=("generation")
# models=("blm" "flm")
models=("blm")

testnum=100

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        output_dir="${task}_${model}_out.json"

        if [ "$model" = "blm" ]; then
            model_name="/data/xunjian_yin/mycode/MAP-NEO/Megatron-LM-NEO/hf_checkpoints/36.70B"
        elif [ "$model" = "flm" ]; then
            model_name="/local/home/yinxunjian/yinxj/mycode/reverse/abductive/FLM/2.8b/step143000/models--EleutherAI--pythia-2.8b-deduped/snapshots/346f515745789fe4b4acbc74b105707cc9d5a36d"
        fi

        if [ "$task" = "generation" ]; then
            python test_anlg.py \
                --model_name $model_name \
                --jsonl_input_file anlg/test-w-comet-preds.jsonl \
                --json_output_file $output_dir \
                --task $task \
                --test_method "demo" \
                --model_type $model \
                --use_gpu \
                --num_samples $testnum \
                --valid_jsonl_file anlg/dev-w-comet-preds.jsonl \
                --num_examples 4
        elif [ "$task" = "classification" ]; then
            python test_anlg.py \
                --model_name $model_name \
                --jsonl_classification_file anli/test.jsonl \
                --txt_labels_file anli/test-labels.lst \
                --json_output_file $output_dir \
                --task $task \
                --test_method "demo" \
                --model_type $model \
                --use_gpu \
                --num_samples $testnum \
                --valid_jsonl_file anlg/dev-w-comet-preds.jsonl \
                --num_examples 4
        fi
    done
done