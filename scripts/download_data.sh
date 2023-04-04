cd data

echo 'downloading argotario'
wget wget https://raw.githubusercontent.com/UKPLab/argotario/master/data/arguments-en-2018-01-15.tsv
mkdir argotario
mv arguments-en-2018-01-15.tsv argotario
python ../../utils/split_and_unify_dataset.py argotario

echo 'downloading propaganda'
wget https://propaganda.qcri.org/nlp4if-shared-task/data/datasets-v2.tgz
tar zxvf datasets-v2.tgz
mv datasets propaganda
python ../../utils/split_and_unify_dataset.py propaganda

echo 'downloading logic'
wget https://raw.githubusercontent.com/causalNLP/logical-fallacy/main/data/edu_train.csv
wget https://raw.githubusercontent.com/causalNLP/logical-fallacy/main/data/edu_dev.csv
wget https://raw.githubusercontent.com/causalNLP/logical-fallacy/main/data/edu_test.csv
mkdir logic
mv edu_*.csv logic
python ../../utils/split_and_unify_dataset.py logic

echo 'all datasets downloaded and split to train, dev, test.\n all datasets saved in data/'