import sys
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report

model_name = "allenai/unifiedqa-v2-t5-3b-1251000" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


data = open(sys.argv[1], 'r').readlines()
data = [json.loads(line) for line in data]

golds, preds = [], []
for i, line in enumerate(data):
	text, label = line['input'], line['target']
	pred = run_model(line['input'])[0]
	print(i)
	print(text)
	print('gold: ', label)
	print('pred: ', pred)
	print()

	golds.append(label)
	preds.append(pred)

print(classification_report(golds, preds))