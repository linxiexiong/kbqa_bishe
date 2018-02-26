import requests
import pandas as pd
import pickle


def fb_mid_wiki_qid(fb2w_file):
    fb_wiki = pd.read_csv(fb2w_file, sep='\t', header=None)
    fb_wiki.columns=['fb', 'rel', 'wiki']
    print (fb_wiki[0:10])
    fb_wiki['fb'] = fb_wiki['fb'].apply(
        lambda x: x.replace("<http://rdf.freebase.com/ns", "").replace(">","").replace(".", "/"))
    fb_wiki['wiki'] = fb_wiki['wiki'].apply(
        lambda x: x.replace("<", "").replace("> .", ""))
    fb_wiki_dict = dict(zip(fb_wiki.fb, fb_wiki.wiki))
    wiki_fb_dict = dict(zip(fb_wiki.wiki, fb_wiki.fb))
    print ({k: fb_wiki_dict[k] for k in list(fb_wiki_dict)[:20]})
    return fb_wiki_dict, wiki_fb_dict


# for python3
def load_pkl_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f, protocol=2)



