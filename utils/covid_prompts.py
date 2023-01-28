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



def prompts_covid(data_dict, fal_list=elena_fal, fal_def_list=elena_fal_def, exclude_fal=['No Fallacy'], comet_list=[], include_pt1=True, include_pt2=False, include_pt3=True, include_pt_unifiedQA=False):
    prompts_targets = [
    ('''Given the claim below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nClaim: {claim}\n''', '{fallacy}'),
    ('''Given the claim and context below, which of the following fallacies does it have: {fallacies}, or {last_fallacy}?\nClaim: {claim}\nContext: {context}\n''', '{fallacy}'),
    ('''Given the following claim, and definitions. Determine which of the fallacies defined below occurs in the claim.\nDefinitions:{definitions}\nClaim: {claim}\n''',
             '{fallacy}'),
    ('Which fallacy does the following claim have: "{claim}"?\n{fallacies}', '{fallacy}') ]
    
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
                
                pt1, pt2, pt3, pt_unifiedQA = prompts_targets # pt[0] has the prompt, pt[1] has the target
                if include_pt1: prompt_data.append(( pt1[0].format(claim=text, fallacies=fal_str, last_fallacy=fal_list[-1]), pt1[1].format(fallacy=fallacy) ))
                if include_pt2:
                    comet_line = fromComet(comet_list[i], text)
                    prompt_data.append(( pt2[0].format(claim=text,  context=comet_line, fallacies=fal_str, last_fallacy=fal_list[-1]), pt2[1].format(fallacy=fallacy) ))
                if include_pt3: prompt_data.append(( pt3[0].format(claim=text, definitions=fal_def_list), pt3[1].format(fallacy=fallacy) ))
                if include_pt_unifiedQA: 
                    prompt_data.append(( pt_unifiedQA[0].format(claim=text, fallacies=fal_str_unifiedQA), pt_unifiedQA[1].format(fallacy=fallacy) ))
                    
                i += 1

    return prompt_data