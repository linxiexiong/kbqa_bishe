import re
import json
import pymysql
import urllib.request
import urllib.parse
from wikiparser import getMainImage, infoBox

class query(object):
    def __init__(self):
        config = {
            'host':"10.61.2.166",
            'port':3306,
            'user':"zengyutao",
            'password':"zengyutao",
            'db':"wikidata",
            'charset':'utf8mb4'
        }
        self.notable_type_names = json.load(open("/home/jinxiaolong/zyt/kbqa_bishe/datas/zyt/all_notable_types.json","r",encoding="UTF-8"))
        self.property_names = json.load(open("/home/jinxiaolong/zyt/kbqa_bishe/datas/zyt/property_names.json","r",encoding="UTF-8"))
        self.conn = pymysql.connect(**config)
        self.cursor = self.conn.cursor()
        print("mysql connected ...")

    def id2id(self, ToColumnName, TableName, FromColumnName, OriID):
        if(OriID == None):
            return None
        sql = "select %s from %s where %s = '%s';" %(ToColumnName, TableName, FromColumnName, OriID)
        self.cursor.execute(sql)
        res = self.cursor.fetchone()
        if(res == None):
            return None
        return res[0]

    def get_index(self, qid):
        config = {
            "ToColumnName":"Table_index",
            "TableName":"entity_index",
            "FromColumnName":"Entity_id",
            "OriID":qid
        }
        return self.id2id(**config)

    def wikititle2wid(self, title):
        config = {
            "ToColumnName":"en_wid",
            "TableName":"wikititle2wid_en",
            "FromColumnName":"en_wikititle",
            "OriID":title
        }
        return self.id2id(**config)

    def wid2wikititle(self, wid):
        config = {
            "ToColumnName":"en_wikititle",
            "TableName":"wikititle2wid_en",
            "FromColumnName":"en_wid",
            "OriID":wid
        }
        return self.id2id(**config)

    def wikititle2qid(self,title):
        config = {
            "ToColumnName":"qid",
            "TableName":"qid2wikititle_en",
            "FromColumnName":"en_wikititle",
            "OriID":title
        }
        return self.id2id(**config)

    def qid2wikititle(self, qid):
        config = {
            "ToColumnName":"en_wikititle",
            "TableName":"qid2wikititle_en",
            "FromColumnName":"qid",
            "OriID":qid
        }
        return self.id2id(**config)

    def wikititle2mid(self, title):
        config = {
            "ToColumnName":"mid",
            "TableName":"mid2wikititle_en",
            "FromColumnName":"en_wikititle",
            "OriID":title
        }
        return self.id2id(**config)

    def mid2wikititle(self, mid):
        config = {
            "ToColumnName":"en_wikititle",
            "TableName":"mid2wikititle_en",
            "FromColumnName":"mid",
            "OriID":mid
        }
        return self.id2id(**config)

    def mid2qid(self, mid):
        config = {
            "ToColumnName":"qid",
            "TableName":"mid2qid",
            "FromColumnName":"mid",
            "OriID":mid
        }
        return self.id2id(**config)

    def qid2mid(self, qid):
        config = {
            "ToColumnName":"mid",
            "TableName":"mid2qid",
            "FromColumnName":"qid",
            "OriID":qid
        }
        return self.id2id(**config)
   
    def mid2wid(self, mid):
        config = {
            "ToColumnName":"en_wid",
            "TableName":"mid2wid_en",
            "FromColumnName":"mid",
            "OriID":mid
        }
        return self.id2id(**config)
    
    def wid2mid(self, wid):
        config = {
            "ToColumnName":"mid",
            "TableName":"mid2wid_en",
            "FromColumnName":"en_wid",
            "OriID":wid
        }
        return self.id2id(**config)
    
    def qid2wid_en(self, qid):
        config = {
            "ToColumnName":"en_wid",
            "TableName":"qid2wid_en",
            "FromColumnName":"qid",
            "OriID":qid
        }
        return self.id2id(**config)
        
    def qid2wid_zh(self, qid):
        config = {
            "ToColumnName":"zh_wid",
            "TableName":"qid2wid_zh",
            "FromColumnName":"qid",
            "OriID":qid
        }
        return self.id2id(**config)
    
    def wid2qid(self, wid):
        config = {
            "ToColumnName":"qid",
            "TableName":"qid2wid_en",
            "FromColumnName":"en_wid",
            "OriID":wid
        }
        res = self.id2id(**config)
        if(res == None):
            config = {
                "ToColumnName":"qid",
                "TableName":"qid2wid_zh",
                "FromColumnName":"zh_wid",
                "OriID":wid
            }
            return self.id2id(**config)
        else:
            return res 
        
    def qid2name(self, qid):
        config = {
            "ToColumnName":"Entity_name",
            "TableName":"entity_names",
            "FromColumnName":"Entity_id",
            "OriID":qid
        }
        return self.id2id(**config)
    
    def qid2name(self, qid):
        config = {
            "ToColumnName":"Entity_name",
            "TableName":"entity_names",
            "FromColumnName":"Entity_id",
            "OriID":qid
        }
        return self.id2id(**config)
    
    def mid2name(self, mid):
        config = {
            "ToColumnName":"name",
            "TableName":"mid2name",
            "FromColumnName":"mid",
            "OriID":mid
        }
        return self.id2id(**config)
    
    def mid2type(self, mid):
        config = {
            "ToColumnName":"notable_type",
            "TableName":"mid2type",
            "FromColumnName":"mid",
            "OriID":mid
        }
        res = self.id2id(**config)
        if(res == None):
            return None
        else:
            return self.notable_type_names[res]
            
            
    def clearName(self, name):
        index1 = name.find("#")
        if(index1 == -1):
            index2 = name.find(":")
            clear_name = name[index2+1:]
        else:
            index2 = name.find(":")
            clear_name = name[index2+1:index1+1]
            index2 = name.find(":",index1)
            clear_name = clear_name + name[index2+1:]
        return clear_name
    
    ## 对得到的三元组进行处理
    ## TODO:对不同的属性值进行解释
    def processWDTriples(self, triples):
        result = []
        for item in triples:
            head_name = self.clearName(item[0])
            relation_name = self.property_names[item[1]]
            ## 可以跳转，为Qid或Pid
            tail_name =  item[2]
            if(item[4] == 0):
                tail_name = item[2] + ":" + self.clearName(str(self.qid2name(item[2])))
            flag = 1-item[4]
            result.append((head_name,relation_name,tail_name,flag))
        
        return result            
    
    def qid2triples(self, qid):
        index = self.get_index(qid)
        if(index == None):
            return None
        else:
            sql = "select Head_name, Relation_id, Tail, Tail_type, Tail_flag from entity_property_%s where Head_id='%s' and Relation_type='Normal';" %(index,qid)
            self.cursor.execute(sql)
            triples = self.cursor.fetchall()
            return self.processWDTriples(triples)
    
    def processFBTriples(self, triples):
        result = []
        for item in triples:
            result.append((item[0], item[1], item[2], 1))
        return result  
    
    def mid2triples(self, mid):
        sql = "select subject, relation, object from FB5M_triples where subject = '%s';" %(mid)
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        return self.processFBTriples(triples)
        
    ## 用于从P18:image属性中获得图片真实在线链接
    def qid2image(self, qid):
        index = self.get_index(qid)
        if(index == None):
            return None
        
        sql = "select Tail from entity_property_%s where Head_id='%s' and Relation_id='P18';" %(index, qid)
        self.cursor.execute(sql)
        res = self.cursor.fetchone()
        if(res == None):
            return None
        
        length = len("https://commons.wikimedia.org/wiki/File:")
        picname = res[0][length:]
        try:
            newurl = "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(picname)
            page = urllib.request.urlopen(newurl)
            text = page.read().decode("UTF-8")
            pattern = re.compile('https://upload.wikimedia.org/wikipedia/commons/[^"]*')
            imageurl = pattern.findall(text)[0]
            return imageurl
        except:
            return None   
        
    def getSummaryById(self, wid):
        if(wid == None):
            return ""
        url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exchars=400&explaintext=true&exlimit=1&pageids=" + urllib.parse.quote(wid)
        try:
            page = urllib.request.urlopen(url)
            res = json.loads(page.read())
            return res['query']['pages'][wid]['extract']
        except:
            summary = ""
            info = infoBox("https://en.wikipedia.org/wiki?curid=" + str(wid))
            for key, value in info.items():
                summary = summary + key + ": " + value + "; "
            return summary
        
    def getImageById(self, wid):
        if(wid == None):
            return ""
        url = "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&pithumbsize=200&format=json&pageids=" + urllib.parse.quote(wid)
        try:
            page = urllib.request.urlopen(url)
            res = json.loads(page.read())
            return res['query']['pages'][wid]['thumbnail']['source']
        except:
            return ( getMainImage("https://en.wikipedia.org/wiki?curid=" + str(wid)) )['link']
        