bert_dir=gs://pjt-bert-bucket/work/model/webcorpus_200122
vocab_dir=../model/webcorpus_base
data_dir=../data/DDQA-1.0/RC-QA
output_dir=../eval_result/ddqa_webcorpus_base
tpu_name=grpc://10.40.53.154:8470

python3 run_squad.py \
  --bert_config_file=$bert_dir/bert_config.json \
  --vocab_file=$vocab_dir/webcorpus.vocab \
  --model_file=$vocab_dir/webcorpus.model \
  --output_dir=$output_dir \
  --train_file=$data_dir/DDQA-1.0_RC-QA_train.json \
  --predict_file=$data_dir/DDQA-1.0_RC-QA_dev.json \
  --init_checkpoint=$bert_dir/model.ckpt \
  --do_train=True \
  --do_predict=True \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --use_tpu=True \
  --tpu_name=$tpu_name \
  --version_2_with_negative=True \