import sys
import json
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def main():

    data = open(sys.argv[2], 'r').readlines()
    data = [json.loads(line) for line in data]
    device = torch.cuda.current_device()
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    model = T5ForConditionalGeneration.from_pretrained(sys.argv[1])
    model.to(device)
    model.eval()
    
    print('model loaded')
    preds = []
    for i, ex in enumerate(data):
        text = ex['translation']['input']

        test_tokenized = tokenizer.encode_plus(text, return_tensors="pt")
        test_input_ids  = test_tokenized["input_ids"].to(device)
        test_attention_mask = test_tokenized["attention_mask"].to(device)

        beam_outputs = model.generate(input_ids=test_input_ids, attention_mask=test_attention_mask)
        preds.append(tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        
        print(i+1)
        print(text)
        print('Predicted Fallacy: ', preds[-1])
        print()

    pickle.dump(preds, open(sys.argv[2].split('.')[0]+'_predicted_fallacies.pkl', 'wb'))
    print('pickle dumped to '+sys.argv[2]+'_predicted_fallacies.pkl')


if __name__ == '__main__':
    main()




















