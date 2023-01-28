''' Tariq Alhindi Oct 24th, 2022
    This script takes three inputs:
        1. A tarined fallacy recognition model (based on the T5) model. (see paper)
        2. A file with text segments (e.g., sentences), one at each line in jsonl format with a key "text" in each line.
    It outputs three files:
        1. 
        2. 
        3. 
'''

import numpy as np
from tqdm import tqdm
import sys, copy, json, pickle, random

import torch
# from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

scheme = str.lower(sys.argv[3])
assert  scheme in ['argo', 'propa', 'logic', 'misinfo']
prompt = str.lower(sys.argv[4])
assert prompt in ['def', 'list']


fallacy_list = {
    'argo': ['Emotional Language', 'Red Herring', 'Hasty Generalization', 'Ad Hominem', 'Irrelevant Authority'],
    'propa': ['Loaded Language', 'Fear or Prejudice', 'Exaggeration or Minimisation', 'Name Calling or Labeling', 'Black-and-White Fallacy', 'Red Herring', 'Slogans', 'Doubt', 'Causal Oversimplification', 'Flag-Waving', 'Irrelevant Authority', 'Thought-terminating Cliches', 'Reductio ad hitlerum', 'Whataboutism', 'Strawman'],
    'logic': ['Emotional Language', 'Causal Oversimplification', 'Ad Populum', 'Circular Reasoning', 'Red Herring', 'Hasty Generalization', 'Ad Hominem', 'Fallacy of Extension', 'Equivocation', 'Deductive Fallacy', 'Irrelevant Authority', 'Intentional Fallacy', 'Black-and-White Fallacy'],
    'misinfo': ['Cherry Picking', 'Causal Oversimplification', 'Red Herring', 'Strawman', 'Irrelevant Authority', 'Hasty Generalization', 'Vagueness', 'Evading the Burden of Proof', 'False Analogy']
}

fallacy_def = {
    'argo' : 
        [
            'Emotional Language: attempting to arouse non-rational sentiments within the intended audience in order to persuade.',
            'Red Herrring: The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.',
            'Hasty Generalization: A generalization is drawn from a sample which is too small, it is not representative of the population or it is not applicable to the situation if all the variables are taken into account.',
            'Ad Hominem: The opponent attacks a person instead of arguing against the claims that the person has put forward.',
            'Irrelevant Authority: An appeal to authority is made where the authority lacks credibility or knowledge in the discussed matter or the authority is attributed a statement which has been tweaked.',
        ],
    'propa' : 
        [
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
        ],
    'logic': 
        [
            'Emotional Language: attempting to arouse non-rational sentiments within the intended audience in order to persuade.',
            'Ad Hominem: The opponent attacks a person instead of arguing against the claims that the person has put forward.',
            'Hasty Generalization: A generalization is drawn from a sample which is too small, it is not representative of the population or it is not applicable to the situation if all the variables are taken into account.',
            'Black-and-White Fallacy: Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).', 
            'Red Herring: The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.', 
            'Irrelevant Authority: An appeal to authority is made where the authority lacks credibility or knowledge in the discussed matter or the authority is attributed a statement which has been tweaked.', 
            'Ad Populum: A fallacious argument which is based on affirming that something is real or better because the majority thinks so.',
            'Circular Reasoning: A fallacy where the end of an argument comes back to the beginning without having proven itself.',
            'Fallacy of Extension: An argument that attacks an exaggerated or caricatured version of your opponent’s position.', 
            'Equivocation: An argument which uses a key term or phrase in an ambiguous way, with one meaning in one portion of the argument and then another meaning in another portion of the argument.', 
            'Deductive Fallacy: An error in the logical structure of an argument. ',
            'Intentional Fallacy: Some intentional (sometimes subconscious) action/choice to incorrectly support an argument.'
        ],
    'misinfo' : 
        [
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
}

instruction_templates = {
    'argo_list' : 'Given the text below, which of the following fallacies occurs in it: {fallacies}, or {last_fallacy}?\nText: {text}',
    'argo_def' : 'Given the following text and definitions, determine which of the fallacies defined below occurs in the text.\nDefinitions:\n{definitions}\n\nText: {text}',
    
    'propa_list' : 'Given the sentence below, which of the following fallacies occur in the sentence: {fallacies}, or {last_fallacy}?\nSentence: {sentence}',
    'propa_def' : 'Given the following sentence and definitions, determine which of the fallacies defined below occur in the sentence.\nDefinitions:\n{definitions}\n\nSentence: {sentence}',
    
    'logic_list' : 'Given the segment below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nSegment: {segment}',
    'logic_def' : 'Given the following segment and definitions, determine which of the fallacies defined below occur in the segment.\nDefinitions:\n{definitions}\n\nSegment: {segment}',
    
    'misinfo_list' : 'Given the claim below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nClaim: {claim}\n',
    'misinfo_def' : 'Given the following claim, and definitions. Determine which of the fallacies defined below occurs in the claim.\nDefinitions:{definitions}\nClaim: {claim}\n',
}

def make_instruction(input_text):
    if prompt == 'list':
        fal_list = copy.deepcopy(fallacy_list[scheme])
        random.shuffle(fal_list)
        fal_str =', '.join(fal_list[:-1])
        if scheme == 'argo':
            instruction = instruction_templates['argo_list'].format(fallacies=fal_str, last_fallacy=fal_list[-1], text=input_text)
        elif scheme == 'propa':
            instruction = instruction_templates['propa_list'].format(fallacies=fal_str, last_fallacy=fal_list[-1], sentence=input_text)
        elif scheme == 'logic':
            instruction = instruction_templates['logic_list'].format(fallacies=fal_str, last_fallacy=fal_list[-1], segment=input_text)
        else: #scheme == 'misinfo'
            instruction = instruction_templates['misinfo_list'].format(fallacies=fal_str, last_fallacy=fal_list[-1], claim=input_text)
    else:
        fal_def = copy.deepcopy(fallacy_def[scheme])
        random.shuffle(fal_def)
        fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def)])
        if scheme == 'argo':
            instruction = instruction_templates['argo_def'].format(definitions=fal_def_str, text=input_text)
        elif scheme == 'propa':
            instruction = instruction_templates['propa_def'].format(definitions=fal_def_str, sentence=input_text)
        elif scheme == 'logic':
            instruction = instruction_templates['logic_def'].format(definitions=fal_def_str, segment=input_text)
        else: #scheme == 'misinfo'
            instruction = instruction_templates['misinfo_def'].format(definitions=fal_def_str, claim=input_text)

    return instruction

def main():

    data = open(sys.argv[2], 'r').readlines()
    data = [json.loads(line) for line in data]
    device = torch.cuda.current_device()

    tokenizer = T5Tokenizer.from_pretrained(sys.argv[1])
    model = T5ForConditionalGeneration.from_pretrained(sys.argv[1])
    model.to(device)
    model.eval()
    print('model loaded')

    all_losses, gen_outputs = [], []
    for i, ex in enumerate(tqdm(data)):
        
        text = make_instruction(ex['text'])
        test_tokenized = tokenizer.encode_plus(text, return_tensors="pt")
        test_input_ids  = test_tokenized["input_ids"].to(device)

        losses = []
        for f in fallacy_list[scheme]:
            label = tokenizer(f, return_tensors="pt", padding="max_length", max_length=50, truncation=True).input_ids.to(device)
            loss = model(input_ids=test_input_ids, labels=label).loss
            # print(f, loss.item())
            losses.append(loss.item())
        all_losses.append(losses)

        beam_outputs = model.generate(input_ids=test_input_ids, num_beams=5, num_return_sequences=3, temperature=1)
        pred_fal1 = tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pred_fal2 = tokenizer.decode(beam_outputs[1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pred_fal3 = tokenizer.decode(beam_outputs[2], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        gen_outputs.append((pred_fal1, pred_fal2, pred_fal3))


    output_dir = '/home/tariq/prj/fallacy/features/'
    # model_name = sys.argv[1].split('/')[-1] if sys.argv[1][-1] != '/' else sys.argv[1].split('/')[-2]
    data_file_name = sys.argv[2].split('.')[0].split('/')[-1]
    # output_file_name = output_dir+'/'+model_name+'_'+data_file_name
    output_file_name = output_dir+'/'+data_file_name

    with open('{}_{}_{}_fallcies.tsv'.format(output_file_name, scheme, prompt), 'w') as f:
        for beam_fal in gen_outputs:
            beam_fal_str = '\t'.join(beam_fal)
            f.write('{}\n'.format(beam_fal_str))

    if 'liarplus' in data_file_name:
        with open('{}_{}_{}_bert.jsonl'.format(output_file_name, scheme, prompt), 'w') as f:
            for beam_fal, line in zip(gen_outputs, data):
                beam_fal_str = 'These three fallacies are the most propabable in the target statement: {}, {}, and {}.'.format(beam_fal[0], beam_fal[1], beam_fal[2])
                f.write('{}\n'.format(json.dumps({'text1': line['text'], 'text2': beam_fal_str, 'label': line['label']})))

        all_losses = np.array(all_losses)
        pickle.dump(all_losses, open('{}_{}_{}_losses.pkl'.format(output_file_name, scheme, prompt), 'wb'))


if __name__ == '__main__':
    main()







