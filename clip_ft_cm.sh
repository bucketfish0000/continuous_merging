SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODEL=openai/clip-vit-base-patch32
MODEL_SHORT_NAME=ViT-B-32

TASK_12=eurosat_stanford
TASK_21=stanford_eurosat

TASK_1=eurosat
TASK_2=stanford-car

fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=classification/clip_finetune \
    method.num_steps=4000 \
    method.save_interval=2000 \
    method.learning_rate=1e-5 \
    modelpool=clip-finetune_${TASK_12}\
    fabric.devices=1 \
    fabric.loggers.root_dir=${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME} \
    fabric.loggers.name=${TASK_12}

python fusion_bench/fusion_bench/scripts/clip/convert_checkpoint.py \
    --checkpoint ${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_12}/version_0/checkpoints/step=3999.ckpt \
    --output ${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_12}/final_model

fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=classification/${TASK_21} \
        method.num_steps=4000 \
        method.save_interval=2000 \
        method.learning_rate=1e-5 \
        modelpool=clip-finetune_${TASK+21} \
        fabric.devices=1 \
        fabric.loggers.root_dir=${SCRIPT_DIR}/output/${MODEL_SHORT_NAME}/${TASK_21} \
        fabric.loggers.name=${TASK_21}

python fusion_bench/fusion_bench/scripts/clip/convert_checkpoint.py \
    --checkpoint ${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_21}/version_0/checkpoints/step=3999.ckpt \
    --output ${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_21}/final_model



fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    +modelpool.modelpool.models._pretrained_.pretrained_model_name_or_path="${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_12}/final_model" \
    taskpool=clip-vit-single-task_${DATA_1}

fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    +modelpool.modelpool.models._pretrained_.pretrained_model_name_or_path="${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_12}/final_model" \
    taskpool=clip-vit-single-task_${DATA_2}

fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    +modelpool.modelpool.models._pretrained_.pretrained_model_name_or_path="${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_21}/final_model" \
    taskpool=clip-vit-single-task_${DATA_1}

fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    +modelpool.modelpool.models._pretrained_.pretrained_model_name_or_path="${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/${TASK_21}/final_model" \
    taskpool=clip-vit-single-task_${DATA_2}



