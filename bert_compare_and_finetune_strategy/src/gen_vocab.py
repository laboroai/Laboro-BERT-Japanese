import os

root_dir = '/home/ubuntu/bert_compare/model/bert-wiki-ja'
ori_vocab_path = os.path.join(root_dir,'wiki-ja.vocab')
output_vocab_path = os.path.join(root_dir,'vocab.txt')

ori_vocab = open(ori_vocab_path,encoding='utf8').readlines()
with open(output_vocab_path,'w',encoding='utf8') as output:
  for n in range(len(ori_vocab)):
    sp_token = ori_vocab[n].strip().split('\t')[0]
    output_token = sp_token
    output.write(output_token + '\n')
