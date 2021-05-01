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
    "training/news-commentary-v13.zh-en"
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
lang=en-zh
prep=wmt18_en_zh
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
    perl $TOKENIZER -threads 64 -a -l $l > $tmp/test.$l
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
newstest2017-enzh-src.en.sgm

python - <<HERE
import random

with open('wmt18_en_zh/tmp/train.en', 'r') as f:
    src = f.readlines()

with open('wmt18_en_zh/tmp/train.zh', 'r') as f:
    tgt = f.readlines()

c = list(zip(src, tgt))
random.shuffle(c)
a, b = zip(*c)

# create 500k dataset
with open('wmt18_en_zh/tmp/train500k.en', 'w') as f:
    for line in a[:750000]:
        f.write(line)

with open('wmt18_en_zh/tmp/train500k.zh', 'w') as f:
    for line in b[:750000]:
        f.write(line)

# create 1M dataset
with open('wmt18_en_zh/tmp/train1M.en', 'w') as f:
    for line in a[:1500000]:
        f.write(line)

with open('wmt18_en_zh/tmp/train1M.zh', 'w') as f:
    for line in b[:1500000]:
        f.write(line)

# create 3M dataset
with open('wmt18_en_zh/tmp/train3M.en', 'w') as f:
    for line in a[:4000000]:
        f.write(line)

with open('wmt18_en_zh/tmp/train3M.zh', 'w') as f:
    for line in b[:4000000]:
        f.write(line)
HERE

TRAIN=$tmp/train.zh-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train3M.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train500k.$L train1M.$L train3M.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

for L in $src $tgt; do
    for f in valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train500k $src $tgt $prep/train500k 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.train1M $src $tgt $prep/train1M 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.train3M $src $tgt $prep/train3M 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

python - <<HERE
# create 500k dataset
with open('wmt18_en_zh/train500k.en', 'r') as f:
    src = f.readlines()

with open('wmt18_en_zh/train500k.zh', 'r') as f:
    tgt = f.readlines()

with open('wmt18_en_zh/train500k.en', 'w') as f:
    for line in src[:500000]:
        f.write(line)

with open('wmt18_en_zh/tmp/train500k.zh', 'w') as f:
    for line in tgt[:500000]:
        f.write(line)

# create 1M dataset
with open('wmt18_en_zh/train1M.en', 'r') as f:
    src = f.readlines()

with open('wmt18_en_zh/train1M.zh', 'r') as f:
    tgt = f.readlines()

with open('wmt18_en_zh/train1M.en', 'w') as f:
    for line in src[:1000000]:
        f.write(line)

with open('wmt18_en_zh/tmp/train1M.zh', 'w') as f:
    for line in tgt[:1000000]:
        f.write(line)

# create 3M dataset
with open('wmt18_en_zh/train3M.en', 'r') as f:
    src = f.readlines()

with open('wmt18_en_zh/train3M.zh', 'r') as f:
    tgt = f.readlines()

with open('wmt18_en_zh/train3M.en', 'w') as f:
    for line in src[:3000000]:
        f.write(line)

with open('wmt18_en_zh/tmp/train3M.zh', 'w') as f:
    for line in tgt[:3000000]:
        f.write(line)
HERE

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done