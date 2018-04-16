source config

reconWeight=1

cut -d ' ' -f 1 data/$config/vocab-freq.$lang1 > data/$config/vocab.$lang1
python3 scripts/translate.py data/$config/transformed-1.$lang1 data/$config/word2vec.$lang2 data/$config/vocab.$lang1 data/$config/result.1
python3 evaluation/evaluate.py evaluation/ldc_cedict.txt data/$config/result.1
