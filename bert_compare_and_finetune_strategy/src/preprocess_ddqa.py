import argparse
import json
import sys
import unicodedata
import os

from tokenizers import TOKENIZERS

EXPES=[
    ("laboro", "jumandic"),
    ("laboro", "laborospm"),
    ("kikuta", "kikutaspm"),
    ("ukyoto", "jumanpp"),
    ("nict", "jumandic"),
    ("utohoku", "neologd")
]

def preprocess(input_path, output_path, tokenize_func):
    with open(input_path) as f:
        data = json.load(f)['data']

    paragraphs = []
    for paragraph in data[0]['paragraphs']:
        context = tokenize_func(paragraph["context"])
        context_strip, offsets = zip(*[(ch, ptr) for ptr, ch in enumerate(context) if not ch.isspace() and ch != "▁"])

        qas = []
        for qa in paragraph['qas']:
            answers = []
            question_tokenized = tokenize_func(qa['question'])
            for answer in qa['answers']:
                answer_strip = "".join(ch for ch in tokenize_func(answer["text"]) if not ch.isspace() and ch != "▁")
                idx = "".join(context_strip).index(answer_strip)
                answer_start, answer_end = offsets[idx], offsets[idx + len(answer_strip) - 1]
                answer_tokenized = context[answer_start:answer_end + 1]
                if ''.join(ch for ch in answer_tokenized if not ch.isspace()) != unicodedata.normalize("NFKC", answer['text']).lower():
                    print(f"No match: {answer_tokenized} vs {answer['text']}")
                # assert ''.join(ch for ch in answer_tokenized if not ch.isspace()) == unicodedata.normalize("NFKC", answer['text']), f"No match: {answer_tokenized} vs {answer['text']}"
                
                answers.append({'text': answer_tokenized, 'answer_start': answer_start})

            qas.append({
                'id': qa['id'],
                'question': question_tokenized,
                'answers': answers
            })
        
        paragraphs.append({
            'context': context,
            'qas': qas
        })
    
    data = [{
        'title': data[0]['title'],
        'paragraphs': paragraphs
    }]

    with open(output_path, mode='w', encoding='utf-8') as f:
        json.dump({'data': data}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/bert_compare_public/data/ddqa')
    parser.add_argument('--mode', type=str, choices=TOKENIZERS.keys(), default=None)
    parser.add_argument('--expe', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    
    if not (args.mode or args.expe):
        for expe, mode in EXPES:
            tokenize_func = TOKENIZERS[mode]
            output_dir = os.path.join(data_dir,expe+'_'+mode)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for split in ['train', 'dev', 'test']:
                preprocess(
                    os.path.join(data_dir,f'DDQA-1.0/RC-QA/DDQA-1.0_RC-QA_{split}.json'),
                    os.path.join(output_dir,f'{split}.json'),
                    tokenize_func)

    elif args.mode and args.expe:
        tokenize_func = TOKENIZERS[args.mode]
        output_dir = os.path.join(data_dir,args.expe+'_'+args.mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for split in ['train', 'dev', 'test']:
            preprocess(
                os.path.join(data_dir,f'DDQA-1.0/RC-QA/DDQA-1.0_RC-QA_{split}.json'),
                os.path.join(output_dir,f'{split}.json'),
                tokenize_func)
    else:
        raise Exception('Either args.mode or args.expe is missing.')
        
        
if __name__ == '__main__':
    main()
