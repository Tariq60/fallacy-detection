import os
import glob
import sys
import ast
import random
import pickle
import pandas as pd
from collections import defaultdict
from split_and_export_dataset import train_dev_test_split, export_train_dev_test_datasets


# 'given sentence, find fragment and fallacy'
# 'given sentence and fallacy, find fragment'
# 'given longer context, find fallacy of type [y]'   LATER
# 'given fallacy and its defnition, does it occur in sentence [x]? (if yes, find fragment)'   LATER

propa_fal = ['Loaded Language', 'Fear or Prejudice', 'Exaggeration or Minimisation', 'Name Calling or Labeling', 'Black-and-White Fallacy', 'Red Herring', 'Slogans', 
             'Doubt', 'Causal Oversimplification', 'Flag-Waving', 'Irrelevant Authority', 'Thought-terminating Cliches', 'Reductio ad hitlerum', 'Whataboutism', 'Strawman']
propa_fal_def = [
    'Loaded Language: Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.',
    'Fear or Prejudice: Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases the support is built based on preconceived judgements.', 
    'Exaggeration or Minimisation: Either representing something in an excessive manner: making things larger, better, worse (e.g., "the best of the best", "quality guaranteed") or making something seem less important or smaller than it really is (e.g., saying that an insult was just a joke).', 
    'Name Calling or Labeling: Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises.', 
    'Black-and-White Fallacy: Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).', 
    'Red Herring: The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.', 
    'Slogans: A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.', 
    'Doubt: Questioning the credibility of someone or something.',
    "Strawman: When an opponent's proposition is substituted with a similar one which is then refuted in place of the original proposition.",
    'Causal Oversimplification: X is identified as the cause of Y when another factor Z causes both X and Y, the causal relation is flipped, or it is a simple correlation not causation.', 
    'Flag-Waving: Playing on strong national feeling (or to any group; e.g., race, gender, political preference) to justify or promote an action or idea.', 
    'Irrelevant Authority: An appeal to authority is made where the authority lacks credibility or knowledge in the discussed matter or the authority is attributed a statement which has been tweaked.', 
    'Thought-terminating Cliches: Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract attention away from other lines of thought.', 
    'Reductio ad hitlerum: Persuading an audience to disapprove an action or idea by suggesting that the idea is popular with groups hated in contempt by the target audience. It can refer to any person or concept with a negative connotation.', 
    "Whataboutism: A technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument."
]

def prompts_propaganda(
        data_dict, 
        fal_list=propa_fal, fal_def_list=propa_fal_def, 
        exclude_fal=['non-probaganda', 'Repetition', 'Bandwagon', 'Obfuscation or Intentional Vagueness or Confusion'],
        include_pt1=True, include_pt3=True, include_pt_unifiedQA=False):
    
    # ('''Given the following sentence and fallacy, find the fragment of the sentence that has this fallacy type.\nSentence: {sentence}\nFallacy: {fallacy}''', 'Fragment: {fragment}')
    prompts_targets = [
        ('''Given the sentence and fragment below, which of the following fallacies occurs in the fragment: {fallacies}, or {last_fallacy}?\nSentence: {sentence}\nFragment: {fragment}''', '{fallacy}'),
        ('''Given the following sentence, fragment and definitions, determine which of the fallacies defined below occurs in the fragment.\nDefinitions:\n{definitions}\n\nSentence: {sentence}\nFragment: {fragment}''', '{fallacy}'),
        ('Which fallacy does the following sentence have: "{sentence}"?\n{fallacies}', '{fallacy}')
    ]
    
    prompt_data, i = [], 0
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for example in data_dict[fallacy]:
                sent, frag = example[2], example[3]
                
                random.shuffle(fal_list)
                fal_str =', '.join(fal_list[:-1])
                fal_str_unifiedQA = '(A) '+fal_list[0]+' (B) '+fal_list[1]+' (C) '+fal_list[2]+' (D) '+fal_list[3]+' (E) '+fal_list[4]+' (F) '+fal_list[5]+' (G) '+fal_list[6]+' (H) '+fal_list[7]+' (I) '+fal_list[8]+' (J) '+fal_list[9]+' (K) '+fal_list[10]+' (L) '+fal_list[11]+' (M) '+fal_list[12]+' (N) '+fal_list[13]
                random.shuffle(fal_def_list)
                fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def_list)])
                
                pt1, pt3, pt_unifiedQA = prompts_targets
                if include_pt1: 
                    prompt_data.append((
                        pt1[0].format(sentence=sent, fragment=frag, fallacies=fal_str, last_fallacy=fal_list[-1]), 
                        pt1[1].format(fallacy=fallacy)
                    ))
                
                if include_pt3: 
                    prompt_data.append((
                        pt3[0].format(sentence=sent, fragment=frag, definitions=fal_def_str), 
                        pt3[1].format(fallacy=fallacy) 
                    ))
                
                if include_pt_unifiedQA: 
                    prompt_data.append((
                        pt_unifiedQA[0].format(sentence=sent, fallacies=fal_str_unifiedQA), 
                        pt_unifiedQA[1].format(fallacy=fallacy) 
                    ))
                    
                i += 1 

    return prompt_data

def getSentLabelsAndFragments(sent):
    text, labels, fragments = sent.split('\t')
    if labels[0] =='n':
        return text, ['non-probaganda'], ['']
    else:
        labels = ast.literal_eval(labels)
        fragments = ast.literal_eval(fragments[:-1])
        return text, labels, fragments

def read_and_preprocess(processed_files_dir, seed=42):
    # read and preprocess file
    filenames = glob.glob(os.path.join(processed_files_dir, '*'))
    prop_ids = [os.path.basename(filename).replace('article', '').split(".")[0] for filename in filenames]
    prop_frag_short = [open(file).readlines() for file in filenames]
    prop_frag_short = [[getSentLabelsAndFragments(sent) for sent in article] for article in prop_frag_short]
    
    prop_frag_short_dict = defaultdict(list)
    for article_id, article in zip(prop_ids, prop_frag_short):
        for sent_id, (sent, labels, fragments) in enumerate(article):
            for label, fragment in zip(labels, fragments):
                prop_frag_short_dict[label].append((article_id, sent_id, sent, fragment.rstrip()))

    for k, v in prop_frag_short_dict.items():
        prop_frag_short_dict[k] = list(dict.fromkeys(prop_frag_short_dict[k]))
        random.seed(seed)
        random.shuffle(prop_frag_short_dict[k])
    
    return prop_frag_short_dict

def main(processed_files_dir, export_dir, seed=42):
    prop_frag_short_dict = read_and_preprocess(processed_files_dir, seed=seed)

    ptrain, pdev, ptest = train_dev_test_split(
        prop_frag_short_dict, min_test_size=6, #exclude_labels={'Repetition', 'non-probaganda'},
        rephrase_labels={'Appeal_to_fear-prejudice': 'Fear or Prejudice', 'Appeal_to_Authority': 'Irrelevant_Authority', 'Straw_Men': 'Strawman'}
    )
    
    ptrain_prompts = prompts_propaganda(ptrain)
    pdev_prompts = prompts_propaganda(pdev)
    ptest_prompts = prompts_propaganda(ptest)
    print(len(ptrain_prompts), len(pdev_prompts), len(ptest_prompts))

    export_train_dev_test_datasets(
        ptrain_prompts, 
        pdev_prompts, 
        ptest_prompts,
        export_dir,
        seed=seed,
        train_file_name='propaganda_train', 
        dev_file_name='propaganda_dev', 
        test_file_name='propaganda_test'
    )
    

if __name__ == '__main__':
    
    '''run like: python propaganda_prompts.py file_dir export_dir'''
    # file_dir = '/data_files/propaganda/processed/articles/'
    # export_dir = './'

    file_dir = sys.argv[1]
    export_dir = sys.argv[2]

    if len(sys.argv) > 3:
        seed = sys.argv[3]
        main(file_dir, export_dir, seed)
    else:
        main(file_dir, export_dir)
