from features import feature_select
import pandas as pd
import sys
sys.path.append("..")
sys.path.append("../..")


def gen_feature(argv):
    data_file = argv[0]
    feature_file = argv[1]
    label_file = argv[2]

    data = pd.read_csv(data_file)
    features, labels = feature_select(data)
    features.to_csv(feature_file, index=False, encoding='utf8')
    labels.to_csv(label_file, index=False, encoding='utf8')


if __name__ == "__main__":
    gen_feature(sys.argv[1:])
