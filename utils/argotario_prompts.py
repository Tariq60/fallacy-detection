argo_fal = ['Emotional Language', 'Red Herring', 'Hasty Generalization', 'Ad Hominem', 'Irrelevant Authority']
argo_fal_def = [
'Emotional Language: attempting to arouse non-rational sentiments within the intended audience in order to persuade.',
'Red Herrring: The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.',
'Hasty Generalization: A generalization is drawn from a sample which is too small, it is not representative of the population or it is not applicable to the situation if all the variables are taken into account.',
'Ad Hominem: The opponent attacks a person instead of arguing against the claims that the person has put forward.',
'Irrelevant Authority: An appeal to authority is made where the authority lacks credibility or knowledge in the discussed matter or the authority is attributed a statement which has been tweaked.',
]


def prompts_argo(data_dict, fal_list=argo_fal, fal_def_list=argo_fal_def, exclude_fal=['No Fallacy'], comet_list=[], include_pt1=True, include_pt2=False, include_pt3=True, include_pt_unifiedQA=False):
    prompts_targets = [
    ('''Given the question and answer below, which of the following fallacies occurs in the answer: {fallacies}, or {last_fallacy}?\nQuestion: {question}\nAnswer: {answer}''', '{fallacy}'),
    ('''Given the question, answer and context below, which of the following fallacies occurs in the answer: {fallacies}, or {last_fallacy}?\nQuestion: {question}\nAnswer: {answer}\nContext: {context}''', '{fallacy}'),
    ('''Given the following question and answer and definitions, determine which of the fallacies defined below occurs in the answer.\nDefinitions:\n{definitions}\n\nQuestion: {question}\nAnswer: {answer}''',
             '{fallacy}'),
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
                
                pt1, pt2, pt3, pt_unifiedQA = prompts_targets # pt[0] has the prompt, pt[1] has the target
                if include_pt1: prompt_data.append(( pt1[0].format(question=ques, answer=ans, fallacies=fal_str, last_fallacy=fal_list[-1]), pt1[1].format(fallacy=fallacy) ))
                if include_pt2:
                    comet_line = fromComet(comet_list[i], ans)
                    prompt_data.append(( pt2[0].format(question=ques, answer=ans, context=comet_line, fallacies=fal_str, last_fallacy=fal_list[-1]), pt2[1].format(fallacy=fallacy) ))
                if include_pt3: prompt_data.append(( pt3[0].format(question=ques, answer=ans, definitions=fal_def_str), pt3[1].format(fallacy=fallacy) ))
                if include_pt_unifiedQA: 
                    prompt_data.append(( pt_unifiedQA[0].format(question=ques, answer=ans, fallacies=fal_str_unifiedQA), pt_unifiedQA[1].format(fallacy=fallacy) ))
                i += 1

    return prompt_data


def main():
    atrain_prompts = prompts_argo(atrain)


if __name__ == '__main__':
    main()
