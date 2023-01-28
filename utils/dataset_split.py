def train_test_split(dict_labels, test_ratio=0.2, min_test_size=10, exclude_labels=set(), rephrase_labels=dict()):
    train_dict, dev_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    
    for k, v in dict_labels.items():
        
        if k in exclude_labels:
            continue
        
        label = rephrase_labels[k] if k in rephrase_labels else k
        label = label.replace('_', ' ').replace(',', ' or ')
        
        '''for each label 20% of the data is test or 10 examples, whichever is greater'''
        test_size = max(int(test_ratio*len(v)), min_test_size)
        # print(label, test_ratio, min_test_size, test_size, len(v), test_size*3, test_size*3 > len(v))
        
        '''keeping only classes that have at least 3 times the test_size,
            i.e. 30 examples total are needed is the default min_test_size of 10 is used'''
#         if test_size*3 > len(v):
#             continue
        
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
            