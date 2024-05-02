import math
from collections import defaultdict
import json
import random


def train_dev_test_split(dict_labels, test_ratio=0.2, min_test_size=10, exclude_labels=set(), rephrase_labels=dict()):
    train_dict, dev_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    
    for k, v in dict_labels.items():
        
        if k in exclude_labels:
            continue
        
        label = rephrase_labels[k] if k in rephrase_labels else k
        label = label.replace('_', ' ').replace(',', ' or ')
        
        '''for each label 20% of the data is test (if test_ratio is at default value) 
        	or 10 examples (if min_test_size is at default value), whichever is greater'''
        test_size = max(int(test_ratio*len(v)), min_test_size)
        # print(label, test_ratio, min_test_size, test_size, len(v), test_size*3, test_size*3 > len(v))
        
        '''keeping only classes that have at least 3 times the test_size,
            i.e. 30 examples total are needed if the default min_test_size of 10 is used <<< condition abandonded'''
		# if test_size*3 > len(v):
		#	continue
        
        for i, item in enumerate(v):
            if i < test_size:
                test_dict[label].append(item)
            else:
                train_dict[label].append(item)
    
    for key in train_dict.keys():
        # dev_size = int(.8*len(train_dict[key]))
        dev_size = math.floor(.8*len(train_dict[key]))
        dev_dict[key] = train_dict[key][dev_size:]
        train_dict[key] = train_dict[key][:dev_size]
    
    return train_dict, dev_dict, test_dict


def export_train_dev_test_datasets(train_list, dev_list, test_list, folder, seed, train_file_name='train', dev_file_name='dev', test_file_name='test'):
    train, dev, test = train_list, dev_list, test_list
    print(len(train), len(dev), len(test))

    random.seed(seed)
    random.shuffle(train)
    random.shuffle(dev)
    train_json = [json.dumps({'translation' : {'input': tup[0], 'target': tup[1]}}) for tup in train]
    dev_json = [json.dumps({'translation' : {'input': tup[0], 'target': tup[1]}}) for tup in dev]
    test_json = [json.dumps({'translation' : {'input': tup[0], 'target': tup[1]}}) for tup in test]

    with open('{}/{}.json'.format(folder, train_file_name), 'w') as f:
        for line in train_json:
            f.write('{}\n'.format(line))
    with open('{}/{}.json'.format(folder, dev_file_name), 'w') as f:
        for line in dev_json:
            f.write('{}\n'.format(line))
    with open('{}/{}.json'.format(folder, test_file_name), 'w') as f:
        for line in test_json:
            f.write('{}\n'.format(line))