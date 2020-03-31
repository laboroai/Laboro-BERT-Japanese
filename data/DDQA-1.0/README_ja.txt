
	     運転ドメインQAデータセット Version 1.0
	     
							     2019/10/31

概要
----

運転ドメインQAデータセットは、ウェブ上で公開されている運転ドメインのブログ記事を基に構築しており、述語項構造QAデータセット（PAS-QAデータセット）と文章読解QAデータセット（RC-QAデータセット）から構成されています。PAS-QAデータセットは、ガ格、ヲ格及びニ格について省略されている項の先行詞を問う問題であり、ガ格は12,468問、ヲ格は3,151問、ニ格は1,069問作成しました。また、RC-QAデータセットは文章の中から質問に対する答えを抽出する問題であり、20,007問作成しました。これらのQAデータセットの作成には、大規模かつ短期間でデータセットを作成可能なクラウドソーシングを利用しました。QAデータセットの形式はSQuAD2.0と同じです。PAS-QAデータセットのガ格とRC-QAデータセットは、全ての問題について文章中に答えがありますが、PAS-QAデータセットのヲ格とニ格は、一部の問題について文章中に答えが無く解答できないものがあります。データセットの構築方法と文章中に答えが無い問題については、参考文献をご参照ください。以下は、運転ドメインQAデータセットの例です。

- PAS-QAデータセット
文章 :
　著者は以下の文章を書きました。本日お昼頃、梅田方面へ自転車で出かけました。ちょっと大きな交差点に差し掛かりました。自転車にまたがった若い女性が信号待ちしています。その後で私も止まって信号が青になるのを待っていました。
質問 :
　待っていました、の主語は何か？
答え :
　私、著者

- RC-QAデータセット
文章 :
　本日お昼頃、梅田方面へ自転車で出かけました。ちょっと大きな交差点に差し掛かりました。自転車にまたがった若い女性が信号待ちしています。その後で私も止まって信号が青になるのを待っていました。
質問 :
　大きな交差点で自転車にまたがっていた人は？
答え :
　若い女性



注意点
------

本データセットは、過去のある時点でウェブ上に公開されていたブログ記事をクロールし、言語情報を付加したもので、自然言語処理の研究開発に資することを目的として公開します。各ブログ記事から採集しているのは4文のみという断片的な情報であることから、各ブログ記事の著作権者の許諾は得ておらず、URLなどの典拠情報は付与しておりません。データセット中のブログ記事の著作権者が、典拠情報の付与もしくはブログ記事の削除を希望した場合には、データセットを修正しアップデートします。その場合には、ダウンロード時に入力いただいたメールアドレスに連絡しますので、必ず古いバージョンを削除し、アップデートをお願いします。



配布リソースの構成
------------------

配布リソースの構成は以下のとおりです。
　README_ja.txt :　READMEファイル（日本語版）
　README_en.txt :　READMEファイル（英語版）
  PAS-QA-NOM/　：　PAS-QAデータセット（ガ格）
　PAS-QA-ACC/　：　PAS-QAデータセット（ヲ格）
  PAS-QA-DAT/　：　PAS-QAデータセット（ニ格）
  RC-QA/　　　 ：　RC-QAデータセット

各QAデータセットのファイル名は以下のとおりです。
|データセット 	|用途	|ファイル名			|
|PAS-QA（ガ格）	|train	|DDQA-1.0_PAS-QA-NOM_train.json	|
|PAS-QA（ガ格）	|dev	|DDQA-1.0_PAS-QA-NOM_dev.json	|
|PAS-QA（ガ格）	|test	|DDQA-1.0_PAS-QA-NOM_test.json	|
|PAS-QA（ヲ格）	|train	|DDQA-1.0_PAS-QA-ACC_train.json	|
|PAS-QA（ヲ格）	|dev	|DDQA-1.0_PAS-QA-ACC_dev.json	|
|PAS-QA（ヲ格）	|test	|DDQA-1.0_PAS-QA-ACC_test.json	|
|PAS-QA（ニ格）	|train	|DDQA-1.0_PAS-QA-DAT_train.json	|
|PAS-QA（ニ格）	|dev	|DDQA-1.0_PAS-QA-DAT_dev.json	|
|PAS-QA（ニ格）	|test	|DDQA-1.0_PAS-QA-DAT_test.json	|
|RC-QA		|train	|DDQA-1.0_RC-QA_train.json	|
|RC-QA		|dev	|DDQA-1.0_RC-QA_dev.json	|
|RC-QA		|test	|DDQA-1.0_RC-QA_test.json	|

なお、データセットの文字コードはUTF-8です。



QAデータセットの形式
--------------------

本QAデータセットの形式はSQuAD2.0と同じです。SQuAD2.0の問題は、「文章」、「質問」、「答え」の三つ組になっており、「答え」は「文章」の中の一部になっています。一部の問題は、「文章」の中に「答え」が無いなど、答えられない問題になっています。詳細は以下の論文をご参照ください。

Pranav Rajpurkar, Robin Jia, and Percy Liang.
Know what you don’t know: Unanswerable questions for SQuAD,
In ACL2018, pages 784–789.
　https://www.aclweb.org/anthology/P18-2124.pdf

以下に、jsonファイル中のQAデータセットを例示します。
　注）jsonファイル中の"context"は「文章」

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



参考文献
--------

高橋 憲生、柴田 知秀、河原 大輔、黒橋 禎夫
ドメインを限定した機械読解モデルに基づく述語項構造解析
言語処理学会 第25回年次大会 発表論文集 (2019年3月)
　https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/B1-4.pdf
　　※データセットの構築方法について記載

Norio Takahashi, Tomohide Shibata, Daisuke Kawahara and Sadao Kurohashi.
Machine Comprehension Improves Domain-Specific Japanese Predicate-Argument Structure Analysis,
In Proceedings of 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, Workshop MRQA: Machine Reading for Question Answering, 2019.
　https://mrqa.github.io/assets/papers/42_Paper.pdf
　　※データセットの構築方法、文章中に答えが無い問題について記載



更新履歴
--------

Version 1.0 　- 2019/10/31公開



連絡先
------

本データセットに関するご意見、ご質問は nl-resource@nlp.ist.i.kyoto-u.ac.jp宛にお願いいたします。データセットに含まれるブログ記事への典拠情報の付与、ブログ記事の削除などをご希望の場合にもこのメールアドレスにご連絡をお願いします。



----------------------------------------------------------------以上
