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

CORPORA=(
    "vi-en/train"
)
DEV_CORPORA=(
    "vi-en/dev"
)
TEST_CORPORA=(
    "vi-en/test"
)
if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=vi
tgt=en
lang=vi-en
# for lack of a better name
prep=train-vi-en
tmp=$prep/tmp

mkdir -p $tmp $prep

echo "pre-processing train data..."
for l in $src $tgt; do
    # rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        python preprocess_translate.py $f.$l $tmp/train.$l $l
    done
done

echo "pre-processing dev data..."
for l in $src $tgt; do
    for f in "${DEV_CORPORA[@]}"; do
        python preprocess_translate.py $f.$l $tmp/valid.$l $l
        echo ""
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    for f in "${TEST_CORPORA[@]}"; do
        python preprocess_translate.py $f.$l $tmp/test.$l $l
        echo ""
    done
done

python - <<HERE
import random

with open('train-vi-en/tmp/train.en', 'r') as f:
    src = f.readlines()

with open('train-vi-en/tmp/train.vi', 'r') as f:
    tgt = f.readlines()

c = list(zip(src, tgt))
random.shuffle(c)
a, b = zip(*c)

# create 500k dataset
with open('train-vi-en/tmp/train500k.en', 'w') as f:
    for line in a[:750000]:
        f.write(line)

with open('train-vi-en/tmp/train500k.vi', 'w') as f:
    for line in b[:750000]:
        f.write(line)

# create 1M dataset
with open('train-vi-en/tmp/train1M.en', 'w') as f:
    for line in a[:1500000]:
        f.write(line)

with open('train-vi-en/tmp/train1M.vi', 'w') as f:
    for line in b[:1500000]:
        f.write(line)

# create 3M dataset
# vi-en has atmost 3.3M data, hence we are not slicing

with open('train-vi-en/tmp/train3M.en', 'w') as f:
    for line in a:
        f.write(line)

with open('train-vi-en/tmp/train3M.vi', 'w') as f:
    for line in b:
        f.write(line)
HERE

TRAIN=$tmp/train.vi-en
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
# # since vi-en has atmost 3M data, we need to check if this is req
perl $CLEAN -ratio 1.5 $tmp/bpe.train3M $src $tgt $prep/train3M 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

python - <<HERE
# create 500k dataset
with open('train-vi-en/train500k.en', 'r') as f:
    src = f.readlines()

with open('train-vi-en/train500k.vi', 'r') as f:
    tgt = f.readlines()

with open('train-vi-en/train500k.en', 'w') as f:
    for line in src[:500000]:
        f.write(line)

with open('train-vi-en/train500k.vi', 'w') as f:
    for line in tgt[:500000]:
        f.write(line)

# create 1M dataset
with open('train-vi-en/train1M.en', 'r') as f:
    src = f.readlines()

with open('train-vi-en/train1M.vi', 'r') as f:
    tgt = f.readlines()

with open('train-vi-en/train1M.en', 'w') as f:
    for line in src[:1000000]:
        f.write(line)

with open('train-vi-en/train1M.vi', 'w') as f:
    for line in tgt[:1000000]:
        f.write(line)

# create 3M dataset
with open('train-vi-en/train3M.en', 'r') as f:
    src = f.readlines()

with open('train-vi-en/train3M.vi', 'r') as f:
    tgt = f.readlines()

with open('train-vi-en/train3M.en', 'w') as f:
    for line in src[:3000000]:
        f.write(line)

with open('train-vi-en/train3M.vi', 'w') as f:
    for line in tgt[:3000000]:
        f.write(line)
HERE

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done