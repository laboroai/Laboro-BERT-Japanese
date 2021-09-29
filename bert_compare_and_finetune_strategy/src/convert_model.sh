CP_PATH=$1

transformers-cli convert --model_type bert \
  --tf_checkpoint $CP_PATH \
  --config $(dirname $CP_PATH)/config.json \
  --pytorch_dump_output $(dirname $CP_PATH)/pytorch_model.bin
