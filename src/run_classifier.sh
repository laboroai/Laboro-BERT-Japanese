bert_dir=../model/webcorpus_base
vocab_dir=../model/webcorpus_base
data_dir=../data/livedoor
output_dir=../eval_result/livedoor_webcorpus_base

python3 run_classifier.py \
	--bert_config_file=$bert_dir/bert_config.json \
	--task_name=livedoor \
	--do_train=true \
	--do_eval=true \
	--data_dir=$data_dir \
	--model_file=$vocab_dir/webcorpus.model \
	--vocab_file=$vocab_dir/webcorpus.vocab \
	--init_checkpoint=$bert_dir/model.ckpt-3900000 \
	--max_seq_length=512 \
	--train_batch_size=4 \
	--learning_rate=2e-5 \
	--num_train_epochs=10 \
	--output_dir=$output_dir