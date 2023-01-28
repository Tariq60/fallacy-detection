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
    fallacies = list(set([ex['translation']['target'] for ex in data]))
    print(fallacies)
    device = torch.cuda.current_device()
    # device = torch.device('cuda')
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    model = T5ForConditionalGeneration.from_pretrained(sys.argv[1])
    model.to(device)
    model.eval()
    
    print('model loaded')
    correct, wrong = 0, 0
    golds, preds_lik, preds_beam, preds_sample, losses = [], [], [], [], []
    for i, ex in enumerate(data):
        text, target = ex['translation']['input'], ex['translation']['target']

        test_tokenized = tokenizer.encode_plus(text, return_tensors="pt")
        # test_tokenized = tokenizer(text, return_tensors="pt")
        test_input_ids  = test_tokenized["input_ids"].to(device)
        test_attention_mask = test_tokenized["attention_mask"].to(device)

        # for f in fallacies:
        #     label = tokenizer(f, return_tensors="pt", padding="max_length", max_length=10, truncation=True).input_ids.to(device)
        #     loss = model(input_ids=test_input_ids, labels=label).loss
        #     print(f, loss)
        #     losses.append([f, loss])
        #     # del label
        #     # torch.cuda.empty_cache()
        # print()
        # preds_lik.append(min(losses, key = lambda t: t[1])[0])

        beam_outputs = model.generate(input_ids=test_input_ids, attention_mask=test_attention_mask)
        preds_beam.append(tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

        # sample_outputs = model.generate(input_ids=test_input_ids, attention_mask=test_attention_mask, do_sample=True, max_length=50, top_k=50)
        # preds_sample.append(tokenizer.decode(sample_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        
        print(i+1); assert len(preds_beam) == i+1
        # print(i+1); assert len(preds_beam) == len(preds_sample) == i+1
        print(text)
        print('target: ', target)
        print('preds_beam: ', preds_beam[-1])
        # print('preds_sample: ', preds_sample[-1])
        print()
        # if target == pred:
        #     correct += 1
        # else:
        #     wrong += 1

        golds.append(target)
        # preds.append(pred)
        # if pred not in fallacies:
        #     print(i, target, pred)
        #     errs.append((i, target, pred))

        # if 'Fallacy:' in target and 'Fallacy:' in pred:
        #     golds.append(target.split('Fallacy:')[1].strip())
        #     preds.append(pred.split('Fallacy:')[1].strip())
        # else:
        #     print('Error in line {} (excluded from evaluation).\n Gold: {}\nPred: {}\n'.format(i, target, pred))
        #     errs.append((i, target, pred))
        


    print(len(data), len(golds), len(preds_lik), len(preds_beam), len(preds_sample))
    # print(correct, wrong)
    # print('loglikelihood'); print(classification_report(golds, preds_lik))
    print('beam'); print(classification_report(golds, preds_beam))
    # print('sample'); print(classification_report(golds, preds_sample))
    
    # cm = confusion_matrix(golds, preds, normalize='all')
    # print(cm)

    # print(sorted(list(set(golds))))
    # cmd = ConfusionMatrixDisplay(cm, display_labels=sorted(list(set(golds+preds))))
    # pickle.dump(cmd, open(sys.argv[2].split('.')[0]+'_cmd.pkl', 'wb'))

    # gpe = {'golds': golds, 'preds': preds, 'errs': errs}
    # pickle.dump(gpe, open(sys.argv[2].split('.')[0]+'_gpe.pkl', 'wb'))

    output_dir = '/home/tariq/prj/fallacy/preds'
    model_name = sys.argv[1].split('/')[-1] if sys.argv[1][-1] != '/' else sys.argv[1].split('/')[-2]
    data_file_name = sys.argv[2].split('.')[0].split('/')[-1]
    output_file_name = output_dir+'/'+model_name+'_'+data_file_name

    with open(output_file_name+'_gold.txt', 'w') as f:
        for line in golds:
            f.write('{}\n'.format(line))
    with open(output_file_name+'_pred.txt', 'w') as f:
        for line in preds_beam:
            f.write('{}\n'.format(line))
    # with open(output_file_name+'_pred_sample.txt', 'w') as f:
    #     for line in preds_sample:
    #         f.write('{}\n'.format(line))




if __name__ == '__main__':
    main()




















