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
BPE_TOKENS=32000

CORPORA=(
    "bg-en/train"
)
DEV_CORPORA=(
    "bg-en/dev"
)
TEST_CORPORA=(
    "bg-en/test"
)
if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=bg
tgt=en
lang=bg-en
prep=UN-bg-en
tmp=$prep/tmp

mkdir -p  $tmp $prep

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

bg_CODE=$prep/bg_code
EN_CODE=$prep/en_code

echo "learn_bpe.py on ${$tmp/train.$src}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.$src > $bg_CODE

echo "learn_bpe.py on ${$tmp/train.$tgt}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.$tgt > $EN_CODE

echo "apply_bpe.py on ${$tmp/train.$src}..."
python $BPEROOT/apply_bpe.py -c $bg_CODE < $tmp/train.$src > $tmp/bpe.train.$src

echo "apply_bpe.py on ${$tmp/train.$tgt}..."
python $BPEROOT/apply_bpe.py -c $EN_CODE < $tmp/train.$tgt > $tmp/bpe.train.$tgt

echo "apply_bpe.py on ${$tmp/valid.$src}..."
python $BPEROOT/apply_bpe.py -c $bg_CODE < $tmp/valid.$src > $tmp/bpe.valid.$src

echo "apply_bpe.py on ${$tmp/valid.$tgt}..."
python $BPEROOT/apply_bpe.py -c $EN_CODE < $tmp/valid.$tgt > $tmp/bpe.valid.$tgt

echo "apply_bpe.py on ${$tmp/test.$src}..."
python $BPEROOT/apply_bpe.py -c $bg_CODE < $tmp/test.$src > $tmp/bpe.test.$src

echo "apply_bpe.py on ${$tmp/test.$tgt}..."
python $BPEROOT/apply_bpe.py -c $EN_CODE < $tmp/test.$tgt > $tmp/bpe.test.$tgt


for L in $src $tgt; do
    cp $tmp/bpe.train.$L $prep/train.$L
    cp $tmp/bpe.valid.$L $prep/valid.$L
    cp $tmp/bpe.test.$L $prep/test.$L
done