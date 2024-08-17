set -ex

export CUDA_VISIBLE_DEVICES=$DEVICE

python -u predict.py \
    --test-file=$TEST_INPUT_FILE \
    --output-file=$OUTPUT_FILE \
    --checkpoint=$FT_MODEL_DIR \
