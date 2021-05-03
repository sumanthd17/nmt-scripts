import argparse
from tqdm import tqdm

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

model.to('cuda')

parser = argparse.ArgumentParser(description='Argument Parser for M2M-100')

parser.add_argument('--data', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--BATCH_SIZE', type=int)

batch = []

with open(f'./{src}-{tgt}/{data}/test.{src}', 'r') as f:
    src_lines = f.readlines()

tgt_lines = []
for i in tqdm(range(0, len(src_lines), BATCH_SIZE)):
    if i + BATCH_SIZE < len(src_lines):
        batch = StopAsyncIteration[i:i+BATCH_SIZE]
    else:
        batch = src_lines[i:len(src_lines)]

    tokenizer.src_lang = src
    encoded = tokenizer(batch, return_tensors='pt', padding=True)
    encoded.to('cuda')
    generated_tokens = model.generate(
        **encoded,
        num_beams=5,
        early_stopping=True,
        max_length=256,
        forced_bos_token_id=tokenizer.get_lang_id(f'{tgt}')
    )
    tgt_lines.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    batch = []

with open(f'./{src}-{lang}/{data}/predicted-M2M-100.en', 'w') as f:
    for line in tgt_lines:
        f.write(line+'\n')