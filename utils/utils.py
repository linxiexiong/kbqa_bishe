from data_processing.mysql import MySQL


def get_name_fome_id_mysql(mid, table_name):
    db = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',pw='zengyutao', db_name='wikidata')
    query = "select name from %s where mid = '%s' " % (table_name, mid)
    print query
    name = db.search(query)
    if len(name) >= 1:
        return name[0]
    return None

