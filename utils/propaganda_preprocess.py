import sys
import os
import glob
import pickle
from nltk.tokenize import sent_tokenize


def read_flc(article_folder, label_folder):
    file_list = glob.glob(os.path.join(article_folder, "*.txt"))
    articles_content, articles_id = ([], [])
    for filename in sorted(file_list):
        with open(filename, "r", encoding="utf-8") as f:
            articles_content.append(f.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])
            
    file_list = glob.glob(os.path.join(label_folder, "*.labels"))
    articles_labels, articles_label_id = ([], [])
    for filename in sorted(file_list):
        with open(filename, "r", encoding="utf-8") as f:
            articles_labels.append(f.readlines())
            articles_label_id.append(os.path.basename(filename).split(".")[0][7:])
    
    return articles_content, articles_id, articles_labels, articles_label_id

def read_slc_labels(label_folder):
    file_list = glob.glob(os.path.join(label_folder, "*.labels"))
    articles_labels, articles_label_id = ([], [])
    for filename in sorted(file_list):
        with open(filename, "r", encoding="utf-8") as f:
            articles_labels.append(f.readlines())
            articles_label_id.append(os.path.basename(filename).split(".")[0][7:])
    
    return articles_labels, articles_label_id

def main(input_dir, export_dir):
    train_folder = os.path.join(input_dir, 'train-articles')
    train_slc_labels_folder = os.path.join(input_dir, 'train-labels-SLC')
    train_flc_labels_folder = os.path.join(input_dir, 'train-labels-FLC')

    train_articles, train_articles_ids, train_flc, train_flc_ids = read_flc(train_folder,train_flc_labels_folder)
    train_flc = [sorted([line.rstrip().split('\t') for line in flc], key=lambda x:int(x[2])) for flc in train_flc]

    train_slc, train_slc_ids = read_slc_labels(train_slc_labels_folder)
    train_slc = [[line.rstrip().split('\t') for line in flc] for flc in train_slc]

    articles_with_types = []
    for j in range(len(train_articles)):
        article, a_len = [], 0
        for paragraph in train_articles[j].split('\n'):
            par_start = a_len
            a_len += len(paragraph) + 1
            if len(paragraph) > 0:
                sentences = [sent for sent in sent_tokenize(paragraph)]
                if len(sentences) == 1:
                    article.append((par_start, a_len, sentences[0]))
                else:
                    article.append((par_start, a_len, sentences))
            else:
                article.append((par_start, a_len, ''))
        
        assert len(train_articles[j]) == a_len-1
        assert len(train_slc[j]) == len(article)-1
        
        article_with_types = []
        for i in range(len(train_slc[j])):
            _, sent_id, sent_label = train_slc[j][i]
            start, end, sent = article[i]
            if sent_label != 'propaganda':
                article_with_types.append((sent, sent_label, ''))
            else:
                types, fragments = [], []
                for _, technique, t_start, t_end in train_flc[j]:
                    t_start, t_end = int(t_start), int(t_end)
                    if (t_start >= start and t_end <= end): # \
                    # or (t_start >= start and t_end >= end and t_start < end) \
                    # or (t_start <= start and t_end <= end and t_end > start):
                        types.append(technique)
                        fragments.append(train_articles[j][t_start:t_end])
                article_with_types.append((sent, types, fragments))
        
        articles_with_types.append(article_with_types)


    for article, article_id in zip(articles_with_types, train_articles_ids):
        with open(os.path.join(export_dir, article_id+'.txt'), 'w') as file:
            for line in article:
                file.write('{}\t{}\t{}\n'.format(line[0],line[1], line[2]))



if __name__ == '__main__':

    '''run like: python propaganda_preprocess.py file_dir export_dir'''
    # input_dir = /Propaganda/data/
    # export_dir = '../data/articles_with_propaganda_types_fragments_short/'

    input_dir = sys.argv[1]
    export_dir = sys.argv[2]
    main(input_dir, export_dir)
