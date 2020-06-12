# How to reproduce our train/test/dev data for Livedoor News Corpus


We cannot provide our experimental train/test/dev datasets made from the Livedoor News Corpus dataset due to the License (CC BY-ND 2.1 JP). We alternatively put index files that indicate which text are used in train/test/dev for our fine-tuning task.

Each index file in this directory, {train,test,dev}_indexed.tsv, contains a header line followed by source filenames and their category names.

We also provide a converter code named "create\_corpus\_from\_index.py". Usage is:


`$ create_corpus_from_index.py sourcedir indexfile > output`

where `sourcedir` is the root directory of the Livedoor News Corpus you obtained and `indexfile` is the path of one of the {train,test,dev}_indexed.tsv. Output format is almost same as the index files, but filename is replaced to its original text with some preprocess.

You will reproduce our fine-tuning results using those output files.
