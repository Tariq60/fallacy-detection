import sys
import random
import pickle
import pandas as pd
from collections import defaultdict
from split_and_export_dataset import train_dev_test_split, export_train_dev_test_datasets

argo_fal = ['Emotional Language', 'Red Herring', 'Hasty Generalization', 'Ad Hominem', 'Irrelevant Authority']
argo_fal_def = [
    'Emotional Language: attempting to arouse non-rational sentiments within the intended audience in order to persuade.',
    'Red Herrring: The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.',
    'Hasty Generalization: A generalization is drawn from a sample which is too small, it is not representative of the population or it is not applicable to the situation if all the variables are taken into account.',
    'Ad Hominem: The opponent attacks a person instead of arguing against the claims that the person has put forward.',
    'Irrelevant Authority: An appeal to authority is made where the authority lacks credibility or knowledge in the discussed matter or the authority is attributed a statement which has been tweaked.',
]


def prompts_argo(
        data_dict, 
        fal_list=argo_fal, fal_def_list=argo_fal_def, exclude_fal=['No Fallacy'], 
        include_list_prompt=True, include_def_prompt=True, include_unifiedQA_prompt=False
    ):
    prompts_targets = [
        ('''Given the question and answer below, which of the following fallacies occurs in the answer: {fallacies}, or {last_fallacy}?\nQuestion: {question}\nAnswer: {answer}''', '{fallacy}'),
        ('''Given the following question and answer and definitions, determine which of the fallacies defined below occurs in the answer.\nDefinitions:\n{definitions}\n\nQuestion: {question}\nAnswer: {answer}''', '{fallacy}'),
        ('Which fallacy does the following answer to the question have: "{question}" "{answer}"?\n{fallacies}', '{fallacy}')
    ]
    
    prompt_data, i = [], 0
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for example in data_dict[fallacy]:
                ques, ans = example
                
                random.shuffle(fal_list)
                fal_str =', '.join(fal_list[:-1])
                fal_str_unifiedQA = '(A) '+fal_list[0]+' (B) '+fal_list[1]+' (C) '+fal_list[2]+' (D) '+fal_list[3]+' (E) '+fal_list[4]
                random.shuffle(fal_def_list)
                fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def_list)])
                
                list_prompt, def_prompt, unifiedQA_prompt = prompts_targets
                if include_list_prompt: 
                    prompt_data.append(( 
                        list_prompt[0].format(question=ques, answer=ans, fallacies=fal_str, last_fallacy=fal_list[-1]), 
                        list_prompt[1].format(fallacy=fallacy)
                    ))
                
                if include_def_prompt: 
                    prompt_data.append(( 
                        def_prompt[0].format(question=ques, answer=ans, definitions=fal_def_str), 
                        def_prompt[1].format(fallacy=fallacy) 
                    ))
                
                if include_unifiedQA_prompt: 
                    prompt_data.append(( 
                        unifiedQA_prompt[0].format(question=ques, answer=ans, fallacies=fal_str_unifiedQA), 
                        unifiedQA_prompt[1].format(fallacy=fallacy) 
                    ))
                i += 1

    return prompt_data


def main(file_dir, export_dir, seed=42):
    # read file
    argo = pd.read_csv(file_dir, sep='\t')

    # preprocessing
    fal = []
    for f1, f2, in zip(argo['Intended Fallacy'], argo['Voted Fallacy']):
        if f2 != '-' and f1 != f2:
            fal.append(f2)
        else:
            fal.append(f1)
    
    # create dictionary: key (fallacy name), value (list of fallacy examples)
    fal_content = defaultdict(list)
    for topic, text, fallacy in zip(argo.Topic, argo.Text, fal):
        if not pd.isna(topic) and not pd.isna(text):
            fal_content[fallacy].append((topic, text))
    
    # shuffle and split
    for fallacy in fal_content:
        random.seed(seed)
        random.shuffle(fal_content[fallacy])

    atrain, adev, atest = train_dev_test_split(
        fal_content, 
        exclude_labels={'No Fallacy'},
        rephrase_labels={'Appeal to Emotion': 'Emotional Language'}
    )
    
    # create prompts
    atrain_prompts = prompts_argo(atrain)
    adev_prompts = prompts_argo(adev)
    atest_prompts = prompts_argo(atest)
    print(len(atrain_prompts), len(adev_prompts), len(atest_prompts))


    export_train_dev_test_datasets(
        atrain_prompts, 
        adev_prompts, 
        atest_prompts, 
        export_dir,
        seed=seed,
        train_file_name='argo_train', 
        dev_file_name='argo_dev', 
        test_file_name='argo_test'
    )


if __name__ == '__main__':

    '''run like: python argotario_prompts.py file_dir export_dir'''
    # file_dir = '../data/argotario/arguments-en-2018-01-15.tsv'
    # export_dir = '../../2023MBZUAI_Fallacy/data/'

    file_dir = sys.argv[1]
    export_dir = sys.argv[2]

    if len(sys.argv) > 3:
        seed = sys.argv[3]
        main(file_dir, export_dir, seed)
    else:
        main(file_dir, export_dir)
