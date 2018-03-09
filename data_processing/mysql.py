#import MySQLdb as mdb
import pymysql as mdb
import traceback
import logging


class MySQL(object):
    def __init__(self, ip, port, user, pw, db_name):
        self.connect = mdb.connect(host=ip, port=port, user=user,
                                   passwd=pw, db=db_name, charset='utf8')
        self.cursor = self.connect.cursor()

    def search(self, query):
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result
        except:
            traceback.print_exc()
            self.connect.rollback()

    def search_all(self, query):
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return result
        except:
            traceback.print_exc()
            self.connect.rollback()

    def close(self):
        self.cursor.close()
        self.connect.close()
        logging.INFO("close db success")


def get_mid_to_name_mysql(db_conn, mid):
        table_name = 'mid2name'
        query = "select name from %s where mid = '%s' " % (table_name, mid)
        #print query
        name = db_conn.search(query)
        if name is not None and len(name) >= 1:
            return name[0]
        return None