from __future__ import unicode_literals, print_function, division
from entity_link.features import data_load, mid_type
from data_processing.mysql import MySQL
import sys
sys.path.append('..')
sys.path.append('../..')


def get_mid_to_name_mysql(db_conn, mid):
        table_name = 'mid2name'
        query = "select name from %s where mid = '%s' " % (table_name, mid)
        #print query
        name = db_conn.search(query)
        if name is not None and len(name) >= 1:
            return name[0]
        return None


def gen_type_train_data():
    sq_data_train = data_load('train')
    #sq_data_valid = data_load('valid')
    sq_data_test = data_load('test')

    sq_data_train = get_type(sq_data_train)
    #sq_data_valid = get_type(sq_data_valid)
    sq_data_test = get_type(sq_data_test)

    #cols = ['questions', 'type', 'type_name']
    print (sq_data_train[0:10])
    sq_data_train.to_csv('type_train.csv')
    #sq_data_valid.to_csv('type_valid.csv')
    sq_data_test.to_csv('type_test.csv')



def get_type(df):
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                  pw='zengyutao', db_name='wikidata')
    df['type'] = df.apply(lambda x: mid_type(db_conn, x['subject_id']), axis=1)
    df['type_name'] = df.apply(lambda x: get_mid_to_name_mysql(db_conn, x['type']), axis=1)
    return df

gen_type_train_data()
