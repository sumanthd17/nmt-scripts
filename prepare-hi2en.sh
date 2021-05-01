echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

echo 'Cloning Indic NLP resources...'
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

CORPORA=(
    "en-hi/train"
)
DEV_CORPORA=(
    "dev/dev"
)
TEST_CORPORA=(
    "test/test"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=hi
tgt=en
lang=en-hi
prep=samanantar_en_hi
tmp=$prep/tmp
orig=orig

mkdir -p $tmp $prep

for l in $src $tgt; do
    for f in "${CORPORA[@]}"; do
        python preprocess_translate.py $orig/$f.$l $tmp/train.$l $l
    done
done

for l in $src $tgt; do
    for f in "${DEV_CORPORA[@]}"; do
        python preprocess_translate.py $orig/$f.$l $tmp/valid.$l $l
    done
done

for l in $src $tgt; do
    for f in "${TEST_CORPORA[@]}"; do
        python preprocess_translate.py $orig/$f.$l $tmp/test.$l $l
    done
done

python - <<HERE
import random

with open('samanantar_en_hi/tmp/train.en', 'r') as f:
    src = f.readlines()

with open('samanantar_en_hi/tmp/train.hi', 'r') as f:
    tgt = f.readlines()

c = list(zip(src, tgt))
random.shuffle(c)
a, b = zip(*c)

# create 500k dataset
with open('samanantar_en_hi/tmp/train500k.en', 'w') as f:
    for line in a[:750000]:
        f.write(line)

with open('samanantar_en_hi/tmp/train500k.hi', 'w') as f:
    for line in b[:750000]:
        f.write(line)

# create 1M dataset
with open('samanantar_en_hi/tmp/train1M.en', 'w') as f:
    for line in a[:1500000]:
        f.write(line)

with open('samanantar_en_hi/tmp/train1M.hi', 'w') as f:
    for line in b[:1500000]:
        f.write(line)

# create 3M dataset
with open('samanantar_en_hi/tmp/train3M.en', 'w') as f:
    for line in a[:4000000]:
        f.write(line)

with open('samanantar_en_hi/tmp/train3M.hi', 'w') as f:
    for line in b[:4000000]:
        f.write(line)
HERE

TRAIN=$tmp/train.hi-en
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

## Apply bpe to dev and test data
perl $CLEAN -ratio 1.5 $tmp/bpe.train500k $src $tgt $prep/train500k 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.train1M $src $tgt $prep/train1M 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.train3M $src $tgt $prep/train3M 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

python - <<HERE
# create 500k dataset
with open('samanantar_en_hi/train500k.en', 'r') as f:
    src = f.readlines()

with open('samanantar_en_hi/train500k.hi', 'r') as f:
    tgt = f.readlines()

with open('samanantar_en_hi/train500k.en', 'w') as f:
    for line in src[:500000]:
        f.write(line)

with open('samanantar_en_hi/train500k.hi', 'w') as f:
    for line in tgt[:500000]:
        f.write(line)

# create 1M dataset
with open('samanantar_en_hi/train1M.en', 'r') as f:
    src = f.readlines()

with open('samanantar_en_hi/train1M.hi', 'r') as f:
    tgt = f.readlines()

with open('samanantar_en_hi/train1M.en', 'w') as f:
    for line in src[:1000000]:
        f.write(line)

with open('samanantar_en_hi/train1M.hi', 'w') as f:
    for line in tgt[:1000000]:
        f.write(line)

# create 3M dataset
with open('samanantar_en_hi/train3M.en', 'r') as f:
    src = f.readlines()

with open('samanantar_en_hi/train3M.hi', 'r') as f:
    tgt = f.readlines()

with open('samanantar_en_hi/train3M.en', 'w') as f:
    for line in src[:3000000]:
        f.write(line)

with open('samanantar_en_hi/train3M.hi', 'w') as f:
    for line in tgt[:3000000]:
        f.write(line)
HERE

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done