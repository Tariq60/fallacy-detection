import sys
import json
import argotario_prompts, climate_prompts, covid_prompts, logic_prompts, propaganda_prompts, split_and_export_dataset

def _parse_line(line, dataset, mode, fallacy):
    if dataset == 'argotario':
        if mode == 'single':
            json_line = {'text': '{} {}'.format(line[0], line[1]), 'label': fallacy}
        else: # mode == 'double'
            json_line = {'text1': line[0], 'text2': line[1], 'label': fallacy}
    
    elif dataset == 'propaganda':
        if mode == 'single':
            json_line = {'text': line[2], 'label': fallacy} # sentence is in line[2]
        elif mode == 'single_frag':
            json_line = {'text': line[3], 'label': fallacy} # fargment is in line[3]
        else: # mode == 'double'
            json_line = {'text1': line[2], 'text2': line[3], 'label': fallacy} # fragment is in line[3]
            
    
    elif dataset == 'climate':
        if mode == 'single':
            json_line = {'text': line[0], 'label': fallacy}
        else: # mode == 'double'
            json_line = {'text1': line[0], 'text2': line[1], 'label': fallacy}
        
    elif dataset == 'covid':
        if mode == 'single':
            json_line = {'text': line[1], 'label': fallacy}
        else: # mode == 'double'
            json_line = {'text1': line[1], 'text2': '', 'label': fallacy}
            
    elif dataset == 'logic':
        if mode == 'single':
            json_line = {'text': line, 'label': fallacy}
        else: # mode == 'double'
            json_line = {'text1': line, 'text2': '', 'label': fallacy}
    
    return json_line
        
        
def create_files(data_dict, dataset, filename, mode='single', exclude_fal=['No Fallacy']):
    '''dataset must be in: argo, propaganda, climate, covid, logic.
        mode must be in: single/single_frag (one text input to bert) or double (two text inputs to bert)'''
    
    json_list = []
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for line in data_dict[fallacy]:
                json_line = _parse_line(line, dataset, mode, fallacy)
                json_list.append(json_line)
    
    with open('{}.json'.format(filename), 'w') as f:
        for line in json_list:
            f.write('{}\n'.format(json.dumps(line)))

def read_preprocess_split(file_dir, dataset):
    if dataset == 'argotario':
        fal_content = argotario_prompts.read_and_preprocess(file_dir)
        train, dev, test = split_and_export_dataset.train_dev_test_split(
            fal_content, exclude_labels={'No Fallacy'},
            rephrase_labels={'Appeal to Emotion': 'Emotional Language'}
        )
    
    elif dataset == 'propaganda':
        fal_content = propaganda_prompts.read_and_preprocess(file_dir)
        train, dev, test = split_and_export_dataset.train_dev_test_split(
            fal_content, min_test_size=6, exclude_labels={'Repetition', 'non-probaganda'},
            rephrase_labels={'Appeal_to_fear-prejudice': 'Fear or Prejudice', 'Appeal_to_Authority': 'Irrelevant_Authority', 'Straw_Men': 'Strawman'}
    )
    elif dataset == 'climate':
        fal_content = climate_prompts.read_and_preprocess(file_dir)
        train, dev, test = split_and_export_dataset.train_dev_test_split(
            fal_content, exclude_labels={'other', 'invalid'}, min_test_size=2,
            rephrase_labels={
                'invalid': 'No Fallacy', 'cherry': 'Cherry Picking', 'evad': 'Evading the Burden of Proof', 'analogy': 'False Analogy',
                'hasty': 'Hasty Generalization', 'vagueness': 'Vagueness', 'strawman': 'Strawman', 'herring': 'Red Herring',
                'authority': 'False Authority', 'post': 'Post Hoc', 'cause': 'False Cause'
            }
        )
    
    elif dataset == 'covid':
        fal_content = covid_prompts.read_and_preprocess(file_dir)
        train, dev, test = split_and_export_dataset.train_dev_test_split(
            fal_content, 
            min_test_size=1,
            rephrase_labels={
                'NONE': 'No Fallacy', 'False analogy': 'False Analogy', 'Evading Burden of Proof': 'Evading the Burden of Proof',
                'Post hoc': 'Post Hoc', 'False cause': 'False Cause', 'False Authority': 'False Authority'
            }
        )
    
    elif dataset == 'logic':
        train, dev, test = logic_prompts.read_preprocess_split(file_dir)
    
    return train, dev, test

if __name__ == '__main__':
    
    '''run like: python bert_files.py file_dir export_dir dataset'''
    # file_dir = '../data/climate_change/all fallacies annotated_final_golden_climate_change.xlsx'
    # export_dir = './'
    # dataset = propaganda
    
    file_dir = sys.argv[1]
    export_dir = sys.argv[2]
    dataset = sys.argv[3]
    assert dataset in ['argotario', 'propaganda', 'climate', 'covid', 'logic']
    train, dev, test = read_preprocess_split(file_dir, dataset)

    mode = sys.argv[4] if len(sys.argv) > 4 else 'single'
    create_files(train, export_dir, dataset, 'train', mode)
    create_files(dev, export_dir, dataset, 'dev', mode)
    create_files(test, export_dir, dataset, 'test', mode)