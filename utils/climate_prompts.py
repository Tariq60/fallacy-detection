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


def prompts_climatechange(
        data_dict, 
        fal_list=fallacies, fal_def_list=fallacy_definitions, exclude_fal=['No Fallacy'],
        include_pt1=True, include_pt2=True, include_pt3=True, include_pt4=True, include_pt_unifiedQA=False
    ):
    
    prompts_targets = [
        ('''Given the segment and comment below, which of the following fallacies occurs in the segment: {fallacies}, or {last_fallacy}?\nSegment: {segment}\nComment: {comment}''', '{fallacy}'),
        ('''Given the segment below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nSegment: {segment}''', '{fallacy}'),
        ('''Given the following segment, comment and definitions, determine which of the fallacies defined below occurs in the segment.\nDefinitions:{definitions}\nSegment: {segment}\nComment: {comment}''', '{fallacy}'),
        ('''Given the following segment and definitions, determine which of the fallacies defined below occurs in the segment.\nDefinitions:{definitions}\nSegment: {segment}\n''', '{fallacy}'),
        ('Which fallacy does the following segment have: "{segment}"?\n{fallacies}', '{fallacy}')
    ]
    
    prompt_data, i = [], 0
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for example in data_dict[fallacy]:
                segment, comment = example
                
                random.shuffle(fal_list)
                fal_str =', '.join(fal_list[:-1])
                fal_str_unifiedQA = '(A) '+fal_list[0]+' (B) '+fal_list[1]+' (C) '+fal_list[2]+' (D) '+fal_list[3]+' (E) '+fal_list[4]+' (F) '+fal_list[5]+' (G) '+fal_list[6]+' (H) '+fal_list[7]+' (I) '+fal_list[8]
                random.shuffle(fal_def_list)
                fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def_list)])
                
                pt1, pt2, pt3, pt4, pt_unifiedQA = prompts_targets
                if include_pt1: 
                    prompt_data.append(( 
                        pt1[0].format(segment=segment, comment=comment, fallacies=fal_str, last_fallacy=fal_list[-1]), 
                        pt1[1].format(fallacy=fallacy) 
                    ))
                
                if include_pt2:
                    prompt_data.append(( 
                        pt2[0].format(segment=segment, fallacies=fal_str, last_fallacy=fal_list[-1]), 
                        pt2[1].format(fallacy=fallacy) 
                    ))
                
                if include_pt3:
                    prompt_data.append(( 
                        pt3[0].format(segment=segment, comment=comment, definitions=fal_def_str), 
                        pt3[1].format(fallacy=fallacy) 
                    ))
                
                if include_pt4:
                    prompt_data.append(( 
                        pt4[0].format(segment=segment, definitions=fal_def_str), 
                        pt4[1].format(fallacy=fallacy) 
                    ))
                
                if include_pt_unifiedQA: 
                    prompt_data.append(( 
                        pt_unifiedQA[0].format(segment=segment, fallacies=fal_str_unifiedQA), 
                        pt_unifiedQA[1].format(fallacy=fallacy) 
                    ))
                    
                i += 1

    return prompt_data


def main(file_dir, export_dir, seed=42):
    
    # read and preprocess file
    climate = pd.read_excel(file_dir)
    fallacies= ['cherry','vagueness', 'herring', 'evad', 'other', 'authority', 'strawman', 'analogy', 'cause', 'post', 'hasty', 'none', 'invalid']
    climate_fal_text = defaultdict(set)
    for i, (text, comment, label) in enumerate(zip(climate['Quote '].values, climate.Comment, climate['Golden annotation '].values)):
        added = False
        for f in fallacies:
            if not pd.isna(label):
                if f in label.strip():
                    f = f.replace('none','invalid')
                    climate_fal_text[f].add((text, comment))
                    added=True
        if not(added) and not pd.isna(label) and label.strip() not in ['both-valid', 'both', '?']:
            print(label)
    
    # shuffle, split, and relabel
    for k, v in climate_fal_text.items():
        climate_fal_text[k] = list(climate_fal_text[k])
        random.seed(seed)
        random.shuffle(climate_fal_text[k])
    
    climate_train, climate_dev, climate_test = train_dev_test_split(
        climate_fal_text, exclude_labels={'other'}, min_test_size=2,
        rephrase_labels={
            'invalid': 'No Fallacy', 'cherry': 'Cherry Picking', 'evad': 'Evading the Burden of Proof', 'analogy': 'False Analogy',
            'hasty': 'Hasty Generalization', 'vagueness': 'Vagueness', 'strawman': 'Strawman', 'herring': 'Red Herring',
            'authority': 'False Authority', 'post': 'Post Hoc', 'cause': 'False Cause'
        }
    )

    clitrain_prompts = prompts_climatechange(climate_train)
    clidev_prompts = prompts_climatechange(climate_dev)
    clitest_prompts = prompts_climatechange(climate_test)
    print(len(clitrain_prompts), len(clidev_prompts),  len(clitest_prompts))

    export_train_dev_test_datasets(
        clitrain_prompts, 
        clidev_prompts, 
        clitest_prompts,
        export_dir,
        seed=seed,
        train_file_name='climate_train', 
        dev_file_name='climate_dev', 
        test_file_name='climate_test'
    )
    


if __name__ == '__main__':
    
    '''run like: python climate_prompts.py file_dir export_dir'''
    # file_dir = '../data/climate_change/all fallacies annotated_final_golden_climate_change.xlsx'
    # export_dir = '../../2023MBZUAI_Fallacy/data/'

    file_dir = sys.argv[1]
    export_dir = sys.argv[2]

    if len(sys.argv) > 3:
        seed = sys.argv[3]
        main(file_dir, export_dir, seed)
    else:
        main(file_dir, export_dir)
