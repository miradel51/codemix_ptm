set -ex

export CUDA_VISIBLE_DEVICES=$DEVICE

python -u train.py \
    --train-query=$TRAIN_QUERY_FILENAME \
    --train-label=$TRAIN_LABEL_FILENAME \
    --dev-query=$DEV_QUERY_FILENAME \
    --dev-label=$DEV_LABEL_FILENAME \
    --pretrained-model-path=$CNT_PTM_DIR \
    --save-model-path=$FT_SAVE_MODEL_PATH