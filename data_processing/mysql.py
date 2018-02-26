import MySQLdb as mdb
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

    def close(self):
        self.cursor.close()
        self.connect.close()
        logging.INFO("close db success")
