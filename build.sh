#!/bin/bash

wget http://algo-recruitment-data.s3-website.eu-central-1.amazonaws.com/train.tsv.gz || exit
wget http://algo-recruitment-data.s3-website.eu-central-1.amazonaws.com/test.tsv.gz  || exit
gunzip train.tsv.gz || exit
gunzip test.tsv.gz || exit

sed -i -e "1d" train.tsv
sed -i -e "1d" test.tsv

awk '{print $1}' train.tsv > train/trainids.txt
awk '{print $2}' train.tsv > train/trainyvalues.txt
awk '{out=$4; for(i=5;i<=NF;i++){out=out" "$i}; print out}' train.tsv > train/traindata.tsv

awk '{print $1}' test.tsv > testa/testids.txt
awk '{out=$3; for(i=4;i<=NF;i++){out=out" "$i}; print out}' test.tsv > testa/testdata.tsv

rm train.tsv test.tsv

