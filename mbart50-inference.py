import argparse
from tqdm import tqdm

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

model.to('cuda')

parser = argparse.ArgumentParser(description='Argument Parser for mBART50')

parser.add_argument('--data', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--src_code', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--BATCH_SIZE', type=int)

args = parser.parse_args()
batch = []

data = args.data
src = args.src
src_code = args.src_code
tgt = args.tgt
BATCH_SIZE = args.BATCH_SIZE

with open(f'./{src}-{tgt}/{data}/test.{src}', 'r') as f:
    src_lines = f.readlines()

tgt_lines = []
for i in tqdm(range(0, len(src_lines), BATCH_SIZE)):
    if i + BATCH_SIZE < len(src_lines):
        batch = src_lines[i:i+BATCH_SIZE]
    else:
        batch = src_lines[i:len(src_lines)]

    tokenizer.src_lang = src_code
    encoded = tokenizer(batch, return_tensors='pt', padding=True)
    encoded.to('cuda')
    generated_tokens = model.generate(
        **encoded,
        num_beams=5,
        early_stopping=True,
        max_length=256
    )
    tgt_lines.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    batch = []

with open(f'./{src}-{tgt}/{data}/predicted-mBART50.{tgt}', 'w') as f:
    for line in tgt_lines:
        f.write(line+'\n')