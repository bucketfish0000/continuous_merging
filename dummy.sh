SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODEL=openai/clip-vit-base-patch32
MODEL_SHORT_NAME=ViT-B-32

SUN=sun397
EUROSAT=eurosat
CAR=stanford_cars
KMNIST=kmnist


fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=classification/clip_finetune \
    method.num_steps=1 \
    method.save_interval=1 \
    method.learning_rate=1e-5 \
    modelpool=clip-finetune_eurosat_stanford\
    fabric.devices=1 \
    fabric.loggers.root_dir=${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME} \
    fabric.loggers.name=eurosat_stanford

python ../fusion_bench/fusion_bench/scripts/clip/convert_checkpoint.py \
    --checkpoint ${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/eurosat_stanford/version_0/checkpoints/step=0.ckpt \
    --output ${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/eurosat_stanford/final_model

fusion_bench \
    --config-dir ${SCRIPT_DIR}/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    +modelpool.modelpool.models._pretrained_.pretrained_model_name_or_path="${SCRIPT_DIR}/outputs/${MODEL_SHORT_NAME}/eurosat_stanford/final_model" \
    taskpool=clip-vit-single-task_sun397