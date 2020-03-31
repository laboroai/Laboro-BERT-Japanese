
       Driving Domain QA Datasets (Version 1.0)

							     2019/10/31

Overview
--------

The Driving Domain QA (Question Answering) Datasets were constructed based on driving domain blog posts published on the web. They consist of a Predicate-Argument Structure QA (PAS-QA) dataset and a Reading Comprehension QA (RC-QA) dataset. We constructed a PAS-QA dataset in which a question asks an omitted argument for a predicate. We made 12,468 problems for the ga case (nominative), 3,151 problems for the wo case (accusative) and 1,069 problems for the ni case (dative). We also constructed an RC-QA dataset that consist of 20,007 problems. Each problem consist of a document, a question and an answer that is a span in the document. We constructed the PAS-QA and RC-QA datasets with crowdsourcing because it enabled to create large-scale datasets in a short time. The data format of these QA datasets is the same as SQuAD 2.0. As for the PAS-QA nominative dataset and the RC-QA dataset, every problem has an answer in a document. However, as for the PAS-QA accusative and dative datasets, some problems cannot be answered because there is no answer in a document. Please refer to the references for how to construct these datasets and how to make problems with no answers. Examples of the Driving Domain QA Datasets are shown below.

- PAS-QA dataset
Document :
　私は右車線に移動した。
　(I moved to the right lane.)  
　(Φが) バックミラーを見た。
　((Φ-NOM) saw the rearview mirror.)
Question : 
　“見た”の主語は何か？
　(What is the subject of "saw"?)
Answer :　
　私 
　(I)

- RC-QA dataset
Document :　
　私の車の前をバイクにまたがった警察官が走っていた。
　(A police officer straddling his bike was running in front of my car.)
Question : 　
　警察官は何に乗っていた？
　(What was the police officer riding?)
Answer :
　バイク 
　(his bike)



Notes
-----

These datasets consist of linguistically annotated blog posts that have been made publicly available on the Web at some time. The datasets are released for the purpose of contributing to the research of natural language processing. Since the collected blog posts are fragmentary, i.e., only the four sentences of each blog post, we have not obtained permission from copyright owners of the blog posts and do not provide source information such as URL. If copyright owners of the blog posts request addition of source information or deletion of these blog posts, we will update the datasets. In this case, we will contact the mail address that was provided for download and ask you to delete the old version and update it. 



Distributed files
-----------------

The configuration of distribution resources is as follows:
　README_ja.txt :　README file (Japanese)
　README_en.txt :　README file (English)
  PAS-QA-NOM/　：　PAS-QA dataset (nominative)
　PAS-QA-ACC/　：　PAS-QA dataset (accusative)
  PAS-QA-DAT/　：　PAS-QA dataset (dative)
  RC-QA/　　　 ：　RC-QA dataset

The file names for each QA dataset are as follows:
|dataset 		|use	|file name			|
|PAS-QA（nominative）	|train	|DDQA-1.0_PAS-QA-NOM_train.json	|
|PAS-QA（nominative）	|dev	|DDQA-1.0_PAS-QA-NOM_dev.json	|
|PAS-QA（nominative）	|test	|DDQA-1.0_PAS-QA-NOM_test.json	|
|PAS-QA（accusative）	|train	|DDQA-1.0_PAS-QA-ACC_train.json	|
|PAS-QA（accusative）	|dev	|DDQA-1.0_PAS-QA-ACC_dev.json	|
|PAS-QA（accusative）	|test	|DDQA-1.0_PAS-QA-ACC_test.json	|
|PAS-QA（dative）	|train	|DDQA-1.0_PAS-QA-DAT_train.json	|
|PAS-QA（dative）	|dev	|DDQA-1.0_PAS-QA-DAT_dev.json	|
|PAS-QA（dative）	|test	|DDQA-1.0_PAS-QA-DAT_test.json	|
|RC-QA			|train	|DDQA-1.0_RC-QA_train.json	|
|RC-QA			|dev	|DDQA-1.0_RC-QA_dev.json	|
|RC-QA			|test	|DDQA-1.0_RC-QA_test.json	|

Note that the encoding of the datasets is UTF-8.



QA dataset format
-----------------

The format of these QA datasets is the same as SQuAD 2.0. A problem of SQuAD 2.0 is a triplet of "Document", "Question" and "Answer". "Answer" is a part of "Document". Some problems cannot be answered because there is no "Answer" in "Document". Please refer to the following paper for details.

Pranav Rajpurkar, Robin Jia, and Percy Liang.
Know what you don’t know: Unanswerable questions for SQuAD,
In ACL2018, pages 784–789.
　https://www.aclweb.org/anthology/P18-2124.pdf

An example of a QA dataset in a json file is as follows:

{
    "version": "v2.0",
    "data": [
        {
            "title": "運転ドメイン",
            "paragraphs": [
                {
                    "context": "著者は以下の文章を書きました。本日お昼頃、梅田方面へ
自転車で出かけました。ちょっと大きな交差点に差し掛かりました。自転車にまたがった若い
女性が信号待ちしています。その後で私も止まって信号が青になるのを待っていました。",
                    "qas": [
                        {
                            "id": "55604556390008_00",
                            "question": "待っていました、の主語は何か？",
                            "answers": [
                                {
                                    "text": "私",
                                    "answer_start": 85
                                },
                                {
                                    "text": "著者",
                                    "answer_start": 0
                                }
                            ],
                            "is_impossible": false
                        }
                    ]
                }
            ]
        }
    ]
}

  Note that "Document" is "context" in json files.



References
----------

Norio Takahashi, Tomohide Shibata, Daisuke Kawahara and Sadao Kurohashi.
Predicate-argument structure analysis based on a machine comprehension model in a specific domain,
In Proceedings of the 25th Annual Meeting of Natural Language Processing (in Japanese), 2019.
　https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/B1-4.pdf
　Note that this paper describes how to construct these datasets.

Norio Takahashi, Tomohide Shibata, Daisuke Kawahara and Sadao Kurohashi.
Machine Comprehension Improves Domain-Specific Japanese Predicate-Argument Structure Analysis,
In Proceedings of 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, Workshop MRQA: Machine Reading for Question Answering, 2019.
　https://mrqa.github.io/assets/papers/42_Paper.pdf
　Note that this paper describes how to construct these datasets and how to make problems with no answers.



Update history
--------------

Version 1.0   - Released on 10/31/2019



Contact
-------

If you have any questions or problems about these datasets, please send an email to nl-resource@nlp.ist.i.kyoto-u.ac.jp. If you have a request to add source information or to delete a document in the datasets, please send an email to this mail address.



----------------------------------------------------------------
