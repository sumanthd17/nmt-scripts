export TEXT=samanantar_en_te

export SPLIT=3M
fairseq-preprocess \
    --source-lang te --target-lang en \
    --trainpref $TEXT/train$SPLIT --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir te-en-$SPLIT/data-bin --thresholdtgt 0 --thresholdsrc 0 \
    --workers 64

export SPLIT=500k
fairseq-preprocess \
    --source-lang te --target-lang en \
    --trainpref $TEXT/train$SPLIT --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir te-en-$SPLIT/data-bin --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict te-en-3M/data-bin/dict.te.txt \
    --tgtdict te-en-3M/data-bin/dict.en.txt \
    --workers 64

export SPLIT=1M
fairseq-preprocess \
    --source-lang te --target-lang en \
    --trainpref $TEXT/train$SPLIT --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir te-en-$SPLIT/data-bin --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict te-en-3M/data-bin/dict.te.txt \
    --tgtdict te-en-3M/data-bin/dict.en.txt \
    --workers 64


export SPLIT= # ADD SPLIT HERE
CUDA_VISIBLE_DEVICES= # ADD DEVICE ID HERE \
fairseq-train te-en-$SPLIT/data-bin \
    --source-lang te \
    --target-lang en \
    --max-target-positions=256 \
    --max-target-positions=256 \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 5e-4 --clip-norm 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
    --max-tokens 32768 \
    --update-freq 2 \
    --patience 5 \
    --fp16 \
    --tensorboard-logdir $SPLIT/tensorboard \
    --save-dir $SPLIT/model \
    --skip-invalid-size-inputs-valid-test \
    --wandb-project nmt-clt