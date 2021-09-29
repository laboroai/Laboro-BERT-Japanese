import argparse
import gzip
import json
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


def preprocess(dataset, mode, output_dir):
    train_dataset = [data for data in dataset if data["timestamp"] < "2009"]
    dev_dataset = [data for data in dataset if "2009" <= data["timestamp"] < "2010"]
    test_dataset = [data for data in dataset if "2010" <= data["timestamp"]]

    tokenize_func = TOKENIZERS[mode]
    
    for filename, datasplit in (
            (os.path.join(output_dir,"train-v1.0.json"), train_dataset),
            (os.path.join(output_dir,"dev-v1.0.json"), dev_dataset),
            (os.path.join(output_dir,"test-v1.0.json"), test_dataset)):
        entries = []
        for data in datasplit:
            for i, document in enumerate(data["documents"]):
                q_id = "{}{:04d}".format(data["qid"], i + 1)
                question = tokenize_func(data["question"])
                answer = "".join(ch for ch in tokenize_func(data["answer"]) if not ch.isspace() and ch != "▁")
                context = tokenize_func(document["text"])
                is_impossible = document["score"] < 2
                if not is_impossible:
                    context_strip, offsets = zip(*[(ch, ptr) for ptr, ch in enumerate(context) if not ch.isspace() and ch != "▁"])
                    idx = "".join(context_strip).index(answer)
                    answer_start, answer_end = offsets[idx], offsets[idx + len(answer) - 1]
                    answer = context[answer_start:answer_end + 1]

                entries.append({
                    "title": q_id,
                    "paragraphs": [{
                        "context": context,
                        "qas": [{
                            "id": q_id,
                            "question": question,
                            "is_impossible": is_impossible,
                            "answers": [{"text": answer, "answer_start": answer_start}] if not is_impossible else []
                        }]
                    }]
                })

        with open(filename, "w", encoding="utf-8") as fp:
            json.dump({"data": entries}, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/bert_compare_public/data/rcqa')
    parser.add_argument('--mode', type=str, choices=TOKENIZERS.keys(), default=None)
    parser.add_argument('--expe', type=str, default=None)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    dataset = []
    with gzip.open(os.path.join(data_dir,"all-v1.0.json.gz"), "rt", encoding="utf-8") as fp:
        for line in fp:
            data = json.loads(line)
            if data["documents"]:
                dataset.append(data)
                
    
    if not (args.mode or args.expe):
        for expe, mode in EXPES:
            output_dir = os.path.join(data_dir, expe+'_'+mode)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            preprocess(dataset, mode, output_dir)

    elif args.mode and args.expe:
        output_dir = os.path.join(data_dir, args.expe+'_'+args.mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        preprocess(dataset, args.mode, output_dir)
    else:
        raise Exception('Either args.mode or args.expe is missing.')


if __name__ == '__main__':
    main()

    