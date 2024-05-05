import sys
import random
import pickle
import pandas as pd
from collections import defaultdict
from split_and_export_dataset import train_dev_test_split, export_train_dev_test_datasets

# definitions for fallacies in climatechange and covid
fallacies = ['Cherry Picking', 'Causal Oversimplification', 'Red Herring', 'Strawman', 'Irrelevant Authority', 'Hasty Generalization', 'Vagueness', 'Evading the Burden of Proof', 'False Analogy']
fallacy_definitions = [
    'Cherry Picking: The act of choosing among competing evidence that which supports a given position, ignoring or dismissing findings which do not support it.',
    'Red Herring: The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.',
    'Irrelevant Authority: An appeal to authority is made where the authority lacks credibility or knowledge in the discussed matter or the authority is attributed a statement which has been tweaked.',
    "Strawman: When an opponent's proposition is substituted with a similar one which is then refuted in place of the original proposition.",
    'Causal Oversimplification: X is identified as the cause of Y when another factor Z causes both X and Y, the causal relation is flipped, or it is a simple correlation not causation.',
    'False Analogy: because two things [or situations] are alike in one or more respects, they are necessarily alike in some other respect.',
    'Hasty Generalization: A generalization is drawn from a sample which is too small, it is not representative of the population or it is not applicable to the situation if all the variables are taken into account.',
    'Vagueness: A word/a concept or a sentence structure which are ambiguous are shifted in meaning in the process of arguing or are left vague being potentially subject to skewed interpretations.',
    'Evading the Burden of Proof: A position is advanced without any arguments supporting it as if it was self-evident.'
]



def prompts_covid(
        data_dict, 
        fal_list=fallacies, fal_def_list=fallacy_definitions, exclude_fal=['No Fallacy'], 
        include_pt1=True, include_pt3=True, include_pt_unifiedQA=False
    ):
    prompts_targets = [
        ('''Given the claim below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nClaim: {claim}\n''', '{fallacy}'),
        ('''Given the following claim, and definitions. Determine which of the fallacies defined below occurs in the claim.\nDefinitions:{definitions}\nClaim: {claim}\n''', '{fallacy}'),
        ('Which fallacy does the following claim have: "{claim}"?\n{fallacies}', '{fallacy}'),
    ]
    
    prompt_data, i = [], 0
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for example in data_dict[fallacy]:
                claim_type, text = example
                
                random.shuffle(fal_list)
                fal_str =', '.join(fal_list[:-1])
                fal_str_unifiedQA = '(A) '+fal_list[0]+' (B) '+fal_list[1]+' (C) '+fal_list[2]+' (D) '+fal_list[3]+' (E) '+fal_list[4]+' (F) '+fal_list[5]+' (G) '+fal_list[6]+' (H) '+fal_list[7]+' (I) '+fal_list[8]
                random.shuffle(fal_def_list)
                fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def_list)])
                
                pt1, pt3, pt_unifiedQA = prompts_targets
                if include_pt1: 
                    prompt_data.append(( 
                        pt1[0].format(claim=text, fallacies=fal_str, last_fallacy=fal_list[-1]), 
                        pt1[1].format(fallacy=fallacy)
                    ))
                
                if include_pt3: 
                    prompt_data.append(( 
                        pt3[0].format(claim=text, definitions=fal_def_str), 
                        pt3[1].format(fallacy=fallacy) 
                    ))
                
                if include_pt_unifiedQA: 
                    prompt_data.append((     
                        pt_unifiedQA[0].format(claim=text, fallacies=fal_str_unifiedQA), 
                        pt_unifiedQA[1].format(fallacy=fallacy)
                    ))
                    
                i += 1

    return prompt_data

def read_and_preprocess(file_dir, seed=42, mode='covid'):
    assert mode in ['covid', 'vaccine']
    fal_content = defaultdict(list)

    if mode == 'covid':
        covid = pd.read_excel(covid_file_dir, header=1)    
        for claim_type, text, fallacy in zip(covid['Type of Claim'], covid.Claim, covid.Fallacy):
            fal_content[fallacy.strip()].append((claim_type, text))
        for k, v in fal_content.items():
            random.seed(seed)
            random.shuffle(fal_content[k]) 
    
    else: # mode == 'vaccine'
        vaccine = pd.read_excel(vaccine_file_dir)
        for claim_type, text, fallacy in zip(vaccine['Type of Claim'], vaccine.Claim, vaccine.Fallacy):
            if not pd.isna(fallacy):
                fallacy = fallacy.replace('ΝΟΝΕ','NONE')
                if ';' not in fallacy: fal_content[fallacy.strip()].append((claim_type, text))
        for k, v in fal_content.items():
            random.seed(seed)
            random.shuffle(fal_content[k])

    return fal_content
    

def main(covid_file_dir, vaccine_file_dir, export_dir, seed=42):
    covid_fal_content = read_and_preprocess(covid_file_dir, seed=seed, mode='covid')
    covid_train, covid_dev, covid_test = train_dev_test_split(
        covid_fal_content, 
        min_test_size=1,
        rephrase_labels={
            'NONE': 'No Fallacy', 'False analogy': 'False Analogy', 'Evading Burden of Proof': 'Evading the Burden of Proof',
            'Post hoc': 'Post Hoc', 'False cause': 'False Cause', 'False Authority': 'False Authority'
        }
    )

    vaccine_fal_content = read_and_preprocess(vaccine_file_dir, seed=seed, mode='vaccine')
    vaccine_train, vaccine_dev, vaccine_test = train_dev_test_split(
        vaccine_fal_content, 
        min_test_size=1,
        rephrase_labels={
            'NONE': 'No Fallacy', 'CP': 'Cherry Picking', 'EBP': 'Evading the Burden of Proof',
            'HG': 'Hasty Generalization', 'VAG': 'Vagueness', 'ST': 'Strawman', 'RH': 'Red Herring',
            'FAUT': 'False Authority', 'PH': 'Post Hoc', 'FC': 'False Cause', 'FA': 'False Analogy'
        }
    )
    covid_train.update(vaccine_train)
    covid_dev.update(vaccine_dev)
    covid_test.update(vaccine_test)

    cvtrain_prompts = prompts_covid(covid_train)
    cvdev_prompts = prompts_covid(covid_dev)
    cvtest_prompts = prompts_covid(covid_test)
    print(len(cvtrain_prompts), len(cvdev_prompts), len(cvtest_prompts))

    export_train_dev_test_datasets(
        cvtrain_prompts, 
        cvdev_prompts, 
        cvtest_prompts,
        export_dir,
        seed=seed,
        train_file_name='covid_train', 
        dev_file_name='covid_dev', 
        test_file_name='covid_test'
    )
    

if __name__ == '__main__':
    
    '''run like: python covid_prompts.py covid_file_dir vaccine_file_dir export_dir'''
    # covid_file_dir = '/data_files/covid/Collected and annotated data_ACL.xlsx'
    # vaccine_file_dir = '/data_files/covid/Collected data Vaccination_Updated_110321 (2).xlsx'
    # export_dir = './'

    covid_file_dir = sys.argv[1]
    vaccine_file_dir = sys.argv[2]
    export_dir = sys.argv[3]

    if len(sys.argv) > 4:
        seed = sys.argv[4]
        main(covid_file_dir, vaccine_file_dir, export_dir, seed)
    else:
        main(covid_file_dir, vaccine_file_dir, export_dir)
