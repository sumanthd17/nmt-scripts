#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz"
    "http://data.statmt.org/wmt18/translation-task/dev.tgz"
    "http://data.statmt.org/wmt18/translation-task/test.tgz"
)
FILES=(
    "training-parallel-nc-v13.tgz"
    "dev.tgz"
    "test.tgz"
)
CORPORA=(
    "training-parallel-nc-v13/news-commentary-v13.zh-en"
    "en-zh/UNv1.0.en-zh"
)
DEV_CORPORA=(
    "dev/newstest2017"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=zh
tgt=en
lang=zh-en
prep=wmt18_zh_en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -7} == ".tar.gz" ]; then
            tar xvf $file
        fi
    fi
done

cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 64 -a -l $l >> $tmp/train.$l
    done
done

echo "pre-processing dev data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2017-zhen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $NORM_PUNC $l | \
	perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 64 -a -l $l > $tmp/valid.$l
    echo ""
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test/newstest2018-zhen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 64 -a -l $l > $tmp/test.$l
    echo ""
done

# TRAIN=$tmp/train.zh-en
# BPE_CODE=$prep/code
# rm -f $TRAIN
# for l in $src $tgt; do
#     cat $tmp/train.$l >> $TRAIN
# done

ZH_CODE=$prep/zh_code
EN_CODE=$prep/en_code

echo "learn_bpe.py on ${$tmp/train.$src}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.$src > $ZH_CODE

echo "learn_bpe.py on ${$tmp/train.$tgt}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.$tgt > $EN_CODE

echo "apply_bpe.py on ${$tmp/train.$src}..."
python $BPEROOT/apply_bpe.py -c $ZH_CODE < $tmp/train.$src > $tmp/bpe.train.$src

echo "apply_bpe.py on ${$tmp/train.$tgt}..."
python $BPEROOT/apply_bpe.py -c $EN_CODE < $tmp/train.$tgt > $tmp/bpe.train.$tgt

echo "apply_bpe.py on ${$tmp/valid.$src}..."
python $BPEROOT/apply_bpe.py -c $ZH_CODE < $tmp/valid.$src > $tmp/bpe.valid.$src

echo "apply_bpe.py on ${$tmp/valid.$tgt}..."
python $BPEROOT/apply_bpe.py -c $EN_CODE < $tmp/valid.$tgt > $tmp/bpe.valid.$tgt

echo "apply_bpe.py on ${$tmp/test.$src}..."
python $BPEROOT/apply_bpe.py -c $ZH_CODE < $tmp/test.$src > $tmp/bpe.test.$src

echo "apply_bpe.py on ${$tmp/test.$tgt}..."
python $BPEROOT/apply_bpe.py -c $EN_CODE < $tmp/test.$tgt > $tmp/bpe.test.$tgt

perl $CLEAN $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done