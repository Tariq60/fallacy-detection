# definitions for fallacies in climatechange and covid
elena_fal = ['Cherry Picking', 'Causal Oversimplification', 'Red Herring', 'Strawman', 'Irrelevant Authority', 'Hasty Generalization', 'Vagueness', 'Evading the Burden of Proof', 'False Analogy']
elena_fal_def = [
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


'''Given the following {arg_type} segment and context. What type of fallacy does it have?'''
'''Given the following segment and fragment, determine the fallacy that occurs in the fragment.'''
'''Given the following segment and {arg_type} sentence, determine the fallacy that occurs in the {arg_type} sentence.'''

def prompts_climatechange(data_dict, fal_list=elena_fal, fal_def_list=elena_fal_def, exclude_fal=['No Fallacy'], comet_list=[],
                          include_pt1=True, include_pt2=True, include_pt3=True, include_pt4=True, include_pt5=False, include_pt6=False, include_pt_unifiedQA=False):
    prompts_targets = [
    ('''Given the segment and comment below, which of the following fallacies occurs in the segment: {fallacies}, or {last_fallacy}?\nSegment: {segment}\nComment: {comment}''', '{fallacy}'),
    ('''Given the segment below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nSegment: {segment}''', '{fallacy}'),
    ('''Given the following segment, comment and definitions, determine which of the fallacies defined below occurs in the segment.\nDefinitions:{definitions}\nSegment: {segment}\nComment: {comment}''',
             '{fallacy}'),
    ('''Given the following segment and definitions, determine which of the fallacies defined below occurs in the segment.\nDefinitions:{definitions}\nSegment: {segment}\n''',
             '{fallacy}'),
    ('''Given the segment, comment and context below, which of the following fallacies occurs in the segment: {fallacies}, or {last_fallacy}?\nSegment: {segment}\nComment: {comment}\nContext: {context}''', '{fallacy}'),
    ('''Given the segment and context below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nSegment: {segment}\nContext: {context}''', '{fallacy}'),
    ('Which fallacy does the following segment have: "{segment}"?\n{fallacies}', '{fallacy}')
    ]
    
    prompt_data, i = [], 0
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for example in data_dict[fallacy]:
                seg, com = example
                
                random.shuffle(fal_list)
                fal_str =', '.join(fal_list[:-1])
                fal_str_unifiedQA = '(A) '+fal_list[0]+' (B) '+fal_list[1]+' (C) '+fal_list[2]+' (D) '+fal_list[3]+' (E) '+fal_list[4]+' (F) '+fal_list[5]+' (G) '+fal_list[6]+' (H) '+fal_list[7]+' (I) '+fal_list[8]
                random.shuffle(fal_def_list)
                fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def_list)])
                
                pt1, pt2, pt3, pt4, pt5, pt6, pt_unifiedQA = prompts_targets # pt[0] has the prompt, pt[1] has the target
                if include_pt1: prompt_data.append(( pt1[0].format(segment=seg, comment=com, fallacies=fal_str, last_fallacy=fal_list[-1]), pt1[1].format(fallacy=fallacy) ))
                if include_pt2: prompt_data.append(( pt2[0].format(segment=seg, fallacies=fal_str, last_fallacy=fal_list[-1]), pt2[1].format(fallacy=fallacy) ))
                if include_pt3: prompt_data.append(( pt3[0].format(segment=seg, comment=com, definitions=fal_def_str), pt3[1].format(fallacy=fallacy) ))
                if include_pt4: prompt_data.append(( pt4[0].format(segment=seg, definitions=fal_def_str), pt4[1].format(fallacy=fallacy) ))
                if include_pt5:
                    comet_line = fromComet(comet_list[i], seg)
                    prompt_data.append(( pt5[0].format(segment=seg, comment=com, context=comet_line, fallacies=fal_str, last_fallacy=fal_list[-1]), pt5[1].format(fallacy=fallacy) ))
                if include_pt6: 
                    comet_line = fromComet(comet_list[i], seg)
                    prompt_data.append(( pt6[0].format(segment=seg, context=comet_line, fallacies=fal_str, last_fallacy=fal_list[-1]), pt6[1].format(fallacy=fallacy) ))
                if include_pt_unifiedQA: 
                    prompt_data.append(( pt_unifiedQA[0].format(segment=seg, fallacies=fal_str_unifiedQA), pt_unifiedQA[1].format(fallacy=fallacy) ))
                    
                i += 1

    return prompt_data