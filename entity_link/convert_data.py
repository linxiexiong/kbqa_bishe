from features import data_load, feature_select, negative_sampling, whole_sample
import pandas as pd
import sys
sys.path.append("..")
sys.path.append("../..")
from data_processing.load_datas import DataReader


def convert_data(argv):
    mid_name_file = "../datas/mid2name.tsv"
    mid_qid_file = "../datas/fb2w.nt"
    print (argv)
    topic_words_file = argv[0]
    sq_data_file = argv[1]
    output_file = argv[2]
    stage = argv[3]
    datas = DataReader(mid_name_file, mid_qid_file,
                       topic_words_file, sq_data_file)
    datas.read_sq_data_pd()
    sq_datas = datas.load_topic_words(stage, datas.sq_dataset)
    sq_datas_whole = whole_sample(sq_datas)
    # sq_datas.fillna(value='')
    sq_datas_whole.to_csv(output_file, index=False, encoding='utf8')


if __name__ == "__main__":
    convert_data(sys.argv[1:])

