# https://github.com/YutaroOgawa/BERT_Japanese_Google_Colaboratory/blob/master/2_BERT_livedoor_news_on_Google_Colaboratory.ipynb
import argparse
from pathlib import Path
import os

import pandas as pd

from tokenizers import TOKENIZERS


EXPES=[
    ("laboro", "jumandic"),
    ("laboro", "laborospm"),
    ("kikuta", "kikutaspm"),
    ("ukyoto", "jumanpp"),
    ("nict", "jumandic"),
    ("utohoku", "neologd")
]


def extract_main_text(path):
    with open(path) as text_file:
        lines = [line.strip() for line in text_file.readlines()[3:]]
        text = ''.join(lines)

        return text

    
def process(data_dir, expe, mode):
    output_dir = os.path.join(data_dir,expe+'_'+mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenize_func = TOKENIZERS[mode]

    xs = []
    ys = []
    for path in Path(data_dir).glob('text/*/*.txt'):
        if path.parts[-1] == 'LICENSE.txt':
            continue
        category = path.parts[-2]
        text = tokenize_func(extract_main_text(path))

        xs.append(text)
        ys.append(category)

    df = pd.DataFrame({'text': xs, 'label': ys})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(df)

    test_ids = [i for i in range(len(df)) if i % 10 == 0]
    dev_ids = [i for i in range(len(df)) if i % 10 == 1]
    train_ids = [i for i in range(len(df)) if i % 10 != 0 and i % 10 != 1]

    df.iloc[test_ids].to_csv(os.path.join(output_dir,'test.tsv'), sep='\t', index=False, header=None)
    df.iloc[dev_ids].to_csv(os.path.join(output_dir,'dev.tsv'), sep='\t', index=False, header=None)
    df.iloc[train_ids].to_csv(os.path.join(output_dir,'train.tsv'), sep='\t', index=False, header=None)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/bert_compare_public/data/ldcc')
    parser.add_argument('--mode', type=str, choices=TOKENIZERS.keys(), default=None)
    parser.add_argument('--expe', type=str, default=None)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not (args.mode or args.expe):
        for expe, mode in EXPES:
            process(data_dir, expe, mode)

    elif args.mode and args.expe:
        process(data_dir, args.expe, args.mode)
    else:
        raise Exception('Either args.mode or args.expe is missing.')

        
if __name__ == '__main__':
    main()
