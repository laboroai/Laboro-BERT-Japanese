import unicodedata

import MeCab
import mojimoji
import pyknp
import sentencepiece as spm

'''
spm_laboro_distilbert = spm.SentencePieceProcessor()
spm_laboro_distilbert.Load("models/laboro_distilbert/tokenizer/ccc_13g_unigram.model")

'''

#tagger_ipadic = MeCab.Tagger("-Owakati")
neologd_dir = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"
tagger_neologd = MeCab.Tagger(f"-Owakati -d{neologd_dir}")

jumanpp = pyknp.Juman("jumanpp")

jumandic_dir = "/usr/local/lib/mecab/dic/jumandic"
tagger_jumandic = MeCab.Tagger(f"-Owakati -d{jumandic_dir}")

spm_laborobert = spm.SentencePieceProcessor()
spm_laborobert.Load("/home/ubuntu/bert_compare_public/model/webcorpus_large_model/webcorpus.model")

spm_bert_wiki_ja = spm.SentencePieceProcessor()
spm_bert_wiki_ja.Load("/home/ubuntu/bert_compare_public/model/bert-wiki-ja/wiki-ja.model")

'''
def tokenize_identity(text):
    return text.replace('\t', ' ')

def tokenize_ipadic(text):
    text = unicodedata.normalize("NFKC", text)
    return tagger_ipadic.parse(text).rstrip("\n")

def tokenize_laboro_distilbert(text):
    return " ".join(spm_laboro_distilbert.EncodeAsPieces(text))


'''

def tokenize_neologd(text):
    return tagger_neologd.parse(text).rstrip('\n')

def tokenize_jumandic(text):
    text = mojimoji.han_to_zen(text).replace('\u3000', ' ')
    return tagger_jumandic.parse(text).rstrip('\n')

def tokenize_jumanpp(text):
    text = mojimoji.han_to_zen(text).replace("\u3000", " ").replace("\n", " ")
    if len(text.encode('utf-8')) > 4096:
        # juman only allows strings up to 4096 bytes. Force turncate if longer
        text = text.encode('utf-8')[:4096].decode('utf-8', errors='ignore')
    return " ".join(mrph.midasi for mrph in jumanpp.analysis(text).mrph_list() if mrph.midasi != "\\ " and mrph.midasi != "\t" and mrph.hinsi)

def tokenize_laborobert(text):
    return " ".join(spm_laborobert.EncodeAsPieces(text))

def tokenize_bert_wiki_ja(text):
    return " ".join(spm_bert_wiki_ja.EncodeAsPieces(text.lower()))

TOKENIZERS = {
    'neologd': tokenize_neologd,
    'jumandic': tokenize_jumandic,
    'jumanpp': tokenize_jumanpp,
    'laborospm': tokenize_laborobert,
    'kikutaspm': tokenize_bert_wiki_ja
}

'''
TOKENIZERS = {
    'identity': tokenize_identity,
    'ipadic': tokenize_ipadic,
    'neologd': tokenize_neologd,
    'jumandic': tokenize_jumandic,
    'bert-wiki-ja': tokenize_bert_wiki_ja,
    'laborobert': tokenize_laborobert,
    'jumanpp': tokenize_jumanpp,
    'laboro_distilbert': tokenize_laboro_distilbert
}
'''
