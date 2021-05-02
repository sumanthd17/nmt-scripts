export lang=te # change language here
export split=3M # change split here
export data=samanantar_en_$lang # change data location here

fairseq-interactive $lang-en/$lang-en-$split/data-bin \
    -s $lang -t en \
    --distributed-world-size 1  \
    --path $lang-en/$split/model/checkpoint_best.pt \
    --batch-size 512 --buffer-size 2500 --beam 5 --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    --input $lang-en/$data/tmp/bpe.test.$lang > $lang-en/outfile-$split.log 2>&1

lines=`wc -l < $lang-en/$data/test.en`

python postprocess_translate.py $lang-en/outfile-$split.log $lang-en/$data/predicted-$split.en $lines en
sacrebleu $lang-en/$data/test.en < $lang-en/$data/predicted-$split.en
