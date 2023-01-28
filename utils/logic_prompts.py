logic_fal = ['Emotional Language', 'Causal Oversimplification', 'Ad Populum', 'Circular Reasoning', 'Red Herring', 'Hasty Generalization', 'Ad Hominem',
             'Fallacy of Extension', 'Equivocation', 'Deductive Fallacy', 'Irrelevant Authority', 'Intentional Fallacy', 'Black-and-White Fallacy']
logic_fal_def = [
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
]


def prompts_logic(data_dict, fal_list=logic_fal, fal_def_list=logic_fal_def, exclude_fal=['No Fallacy'], comet_list=[], include_pt1=True, include_pt2=False, include_pt3=True, include_pt_unifiedQA=False):
    prompts_targets = [
    ('''Given the segment below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nSegment: {segment}''', '{fallacy}'),
    ('''Given the segment and context below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nSegment: {segment}\nContext: {context}''', '{fallacy}'),
    ('''Given the following segment and definitions, determine which of the fallacies defined below occurs in the segment.\nDefinitions:\n{definitions}\n\nSegment: {segment}''',
             '{fallacy}'),
    ('Which fallacy does the following segment have: "{segment}"?\n{fallacies}', '{fallacy}')]
    
    prompt_data, i = [], 0
    for fallacy in data_dict.keys():
        if fallacy not in exclude_fal:
            for example in data_dict[fallacy]:
                
                random.shuffle(fal_list)
                fal_str =', '.join(fal_list[:-1])
                fal_str_unifiedQA = '(A) '+fal_list[0]+' (B) '+fal_list[1]+' (C) '+fal_list[2]+' (D) '+fal_list[3]+' (E) '+fal_list[4]+' (F) '+fal_list[5]+' (G) '+fal_list[6]+' (H) '+fal_list[7]+' (I) '+fal_list[8]+' (J) '+fal_list[9]+' (K) '+fal_list[10]+' (L) '+fal_list[11]+' (M) '+fal_list[12]
                random.shuffle(fal_def_list)
                fal_def_str ='\n'.join([str(i+1)+'. '+s for i, s in enumerate(fal_def_list)])
                
                pt1, pt2, pt3, pt_unifiedQA = prompts_targets # pt[0] has the prompt, pt[1] has the target
                if include_pt1: prompt_data.append(( pt1[0].format(segment=example, fallacies=fal_str, last_fallacy=fal_list[-1]), pt1[1].format(fallacy=fallacy) ))
                if include_pt2:
                    comet_line = fromComet(comet_list[i], example)
                    prompt_data.append(( pt2[0].format(segment=example, context=comet_line, fallacies=fal_str, last_fallacy=fal_list[-1]), pt2[1].format(fallacy=fallacy) ))
                if include_pt3: prompt_data.append(( pt3[0].format(segment=example, definitions=fal_def_str), pt3[1].format(fallacy=fallacy) ))
                if include_pt_unifiedQA: 
                    prompt_data.append(( pt_unifiedQA[0].format(segment=example, fallacies=fal_str_unifiedQA), pt_unifiedQA[1].format(fallacy=fallacy) ))
                    
                i += 1

    return prompt_data