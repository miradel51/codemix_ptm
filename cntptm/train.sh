set -ex

export CUDA_VISIBLE_DEVICES=$DEVICE

python -u train.py \
    --train-query=$TRAIN_QUERY_FILENAME \
    --train-label=$TRAIN_LABEL_FILENAME \
    --pretrained-model-path=$PRETRAINED_MODEL_PATH \
    --save-model-path=$CNT_PTM_SAVE_MODEL_PATH
