#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
# Runs the "NEO-7B" parameter model
SCRIPT_DIR=$(dirname $0)
MEGATRON_DIR=$(realpath ${SCRIPT_DIR}/../../../..)
echo $MEGATRON_DIR
export PYTHONPATH=$PYTHONPATH:$MEGATRON_DIR
echo $PYTHONPATH
# export NCCL_SOCKET_IFNAME=ibp
# export NCCL_IB_HCA=mlx5
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY=5d213494b040070e178f39e1cb3b795e2eeedb92
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_SOCKET_TIMEOUT=3600
# export NCCL_IB_TIMEOUT=3600
# export LOGLEVEL=INFO
# export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export NCCL_DEBUG=WARN


export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-1}
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
# MASTER_ADDR=localhost
MASTER_PORT=8874

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export MASTER_ADDR=222.29.51.158

echo $MASTER_ADDR



export DP_SIZE=$((WORLD_SIZE / PP_SIZE / TP_SIZE))
export MICRO_BATCH=${MICRO_BATCH:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-32}
export GLOBAL_BATCH=$((DP_SIZE * MICRO_BATCH * GRAD_ACC_STEPS))

echo "[pretrain], GPUS_PER_NODE: $GPUS_PER_NODE"
echo "[pretrain], NNODES: $NNODES"
echo "[pretrain], NODE_RANK: $NODE_RANK"
echo "[pretrain], MASTER_ADDR: $MASTER_ADDR"
echo "[pretrain], MASTER_PORT: $MASTER_PORT"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
"

export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-24576}

    # --kv_channels 256 \
NEO_MODELING_ARGS="
    --use-mcore-models \
    --num-layers 28 \
    --hidden-size 3072 \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads 16 \
    --max-position-embeddings 8192 \
    --group-query-attention \
    --num-query-groups ${NUM_KV_HEADS:-16} \
    --swiglu \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --disable-bias-linear
"

NEO_HYPER_PARAM_ARGS="
    --seed ${SEED:-42} \
    --seq-length 8192 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --bf16 \
    --eod-mask-loss \
    --norm-epsilon 1e-5 \
    --lr 2e-4 \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --init-method-std 0.02 \
    --override-opt_param-scheduler
"

NEO_TRAINING_ARGS="
    --num-workers 8 \
    --distributed-backend nccl \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --expert-model-parallel-size ${EP_SIZE:-1} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --optimizer adam \
    --train-iters ${TRAIN_ITERS:-51900} \
    --exit-interval ${EXIT_ITERS:-${TRAIN_ITERS:-51900}}
"

echo "[pretrain], begin..."
echo "[pretrain], WORLD_SIZE: $WORLD_SIZE, GPUS_PER_NODE: $GPUS_PER_NODE, NNODES: $NNODES"
echo "[pretrain], DP_SIZE: $DP_SIZE, TP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE"
echo "[pretrain], Global batch size: $GLOBAL_BATCH, micro batch size: $MICRO_BATCH"
echo "[pretrain], GRAD_ACC_STEPS: $GRAD_ACC_STEPS"

TASK_ID=${TASK_ID:-"rPretrain7b"}
JOB_NAME=NEO_7b_nl${NUM_LAYERS}_tp${TP_SIZE}_pp${PP_SIZE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_gas${GRAD_ACC_STEPS}
OUTPUT_HOME="/data/xunjian_yin/mycode/MAP-NEO/Megatron-LM-NEO/checkpoints/$JOB_NAME/$TASK_ID"
CHECKPOINT_PATH="${OUTPUT_HOME}/checkpoint/"
WANDB_PATH="${OUTPUT_HOME}/"

# export DATA_PATH=$(echo "${DATA_PATH}" | base64 --decode)
export DATA_PATH="259833174808 /data/xunjian_yin/mycode/DCLM/dclm-b1/dclm_reverse/dclm_01_text_document 21951239040 /data/public_models/huggingface/matrix/tmp/rr_code_code.000_text_document 51337646588 /data/public_models/huggingface/matrix/tmp/rr_exam_math_text_document 37370709647 /data/public_models/huggingface/matrix/tmp/rr_paper_math.000_text_document 26294804192 /data/public_models/huggingface/matrix/tmp/rr_code_code.002_text_document 14259302578 /data/public_models/huggingface/matrix/tmp/rr_cc_math.000_text_document 24327029000 /data/xunjian_yin/mycode/DCLM/dclm-b1/dclm_reverse/dclm_02_text_document"
export DATA_CACHE_PATH=${DATA_CACHE_PATH:-"null"}
export TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-"neo/tokenizer.model"}

echo "[pretrain], DATA_PATH: $DATA_PATH"
echo "[pretrain], DATA_CACHE_PATH: $DATA_CACHE_PATH"
echo "[pretrain], TOKENIZER_MODEL_PATH: $TOKENIZER_MODEL_PATH"


export ENABLE_SHUFFLE=${ENABLE_SHUFFLE:-"false"}
shuffle_args=""
if [[ $ENABLE_SHUFFLE == "true" ]]; then
  shuffle_args="--enable-shuffle"
fi
echo "[pretrain], ENABLE_SHUFFLE: $ENABLE_SHUFFLE"

NEO_DATA_ARGS="
    --train-data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    --split 1000,0,0 \
    $shuffle_args
"
load_args=""
if [[ $(ls ${CHECKPOINT_PATH} 2> /dev/null | wc -l ) > 0 ]]; then
  load_args="--load ${CHECKPOINT_PATH}"
fi
CHECKPOINT_ARGS="
    --save $CHECKPOINT_PATH \
    $load_args
"

export WANDB_PROJECT=${WANDB_PROJECT:-"neo_test"}
export WANDB_EXP_NAME=${WANDB_EXP_NAME:-${TASK_ID}_${JOB_NAME}}

WANDB_ARGS="
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${WANDB_EXP_NAME} \
    --wandb-save-dir ${WANDB_PATH} \
"

export SAVE_INTERVAL=${SAVE_INTERVAL:-1125}

NEO_OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-iters 0 \
    --eval-interval 1000000 \
    --timing-log-level=0
"

export PY_SCRIPT_PATH=${PY_SCRIPT_PATH:-neo/pretrain_gpt_neo_test.py}
CMD="torchrun $DISTRIBUTED_ARGS $PY_SCRIPT_PATH \
    $NEO_MODELING_ARGS \
    $NEO_HYPER_PARAM_ARGS \
    $NEO_TRAINING_ARGS \
    $NEO_DATA_ARGS \
    $NEO_OUTPUT_ARGS \
    $CHECKPOINT_ARGS \
    $WANDB_ARGS \
    "

echo "----------------------------------------------------"
echo $CMD
echo "----------------------------------------------------"
$CMD
