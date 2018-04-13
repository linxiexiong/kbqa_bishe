from flask import *
import re
import time
import json
import random
import pandas as pd
import urllib.parse
import urllib.request
import wikiparser as parser
import wikipedia as wiki
from pprint import pprint
from nltk.tokenize import word_tokenize

import util
import integration

QA = integration.QustionAnswering()
myquery = util.query()

app = Flask(__name__, static_folder='static')
app.config['DEBUG'] = True

def get_url(string):
    try:
        tmp = wiki.page(string)
        return tmp.url
    except Exception as e:
        info = e.__str__()
        entities = []
        entities = info.split("\n")[1:-3]
        if(len(entities) == 0):
            return ""
        return wiki.page(entities[0]).url


def get_summary(string):
    try:
        tmp = wiki.page(string)
        return tmp.summary
    except Exception as e:
        info = e.__str__()
        entities = []
        entities = info.split("\n")[1:-3]
        if(len(entities) == 0):
            return ""
        return wiki.page(entities[0]).summary


def get_google_results(query, num):
    keyfile = open("./.api_key","r")
    cx = keyfile.readline()[:-1]
    key = keyfile.readline()[:-1]
    keyfile.close()
    paras = {
        'q':query,
        'num':num,
        'cx':cx,
        'key':key
    }
    url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(paras)
    print(url)
    try:
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
    except:
        return dict({})
    keyfile.close()
    return data['items']


def google_results_clear(result):
    res = []
    for item in result:
        part = {}
        part['title'] = item['title']
        part['url'] = item['link']
        if(part['url'].startswith("http") ==  False):
            part['url'] = "http://" + part['url']
        part['snippet'] = item['snippet']
        if('cse_image' in item['pagemap']):
            part['image'] = item['pagemap']['cse_image'][0]['src']
        else:
            part['image'] = None

        res.append(part)
    return res

def get_mainentity_info(mid, show_threshold):
    qid = myquery.mid2qid(mid)
    name = myquery.mid2name(mid)
    if(qid != None):
        wid = myquery.qid2wid_en(qid)
    else:
        wid = myquery.mid2wid(mid)
    
    time1 = time.time()
    image = myquery.getImageById(wid)
    time2 = time.time()
    print("get mainEntity image time used: " + str(time2-time1) + "s")
    
    ## TODO：这里是否需要使用 Wikidata 的三元组
    time1 = time.time()
    triples = myquery.mid2triples(mid)
    time2 = time.time()
    print("get mainEntity triples time used: " + str(time2-time1) + "s")
    
    if(triples == None): 
        triples = []
    graphdata = []
    
    ## 对应于 Wikidata
#     for tup in triples:
#         for item in tup:
#             graphdata.append(item)

    print("triples num: " + str(len(triples)) + " and mid is: " + str(mid))
    ## 对应于 Freebase
    time1 = time.time()
    count = 0
    for tup in triples:
        count = count + 1
        graphdata.append(name)
        graphdata.append(tup[1])
        if(count <= show_threshold):
            obj_name = myquery.mid2name(tup[2])
        else:
            obj_name = tup[2]
        obj_name = "" if(obj_name == None) else obj_name
        graphdata.append(obj_name)
        graphdata.append(tup[3])
    
    time2 = time.time()
    print("get graphdata time used: " + str(time2-time1) + "s")
    
    time1 = time.time()
    summary = myquery.getSummaryById(wid)
    time2 = time.time()
    print("get mainEntity summary time used: " + str(time2-time1) + "s\n")
    
    return image,summary,graphdata,triples

## 统计string1中单词在string2中出现频率
def word_count(string1, string2):
    count = 0
    list1 = word_tokenize(string1)
    for word in list1:
        count = count + string2.count(word)
    return count    


def get_top_relation(keywords, relations):
    if(relations == None or keywords == None):
        return None
   
    rel_set = set()
    for rel in relations:
        rel_set.add(rel)
    
    rel_scores = {}
    for rel in rel_set:
        ## 如果是Wikidata：仅对英文名计算相似度
        # word_list = word_tokenize(rel.split("#")[0])
        ## 如果是Freebase：需要对每个单词进行处理，并去除开头的 /
        word_list = re.split('[_|/]', rel[1:])
        rel_len = len(word_list)
        rel_scores[rel] = sorted([word_count(x, rel)/rel_len for x in keywords], reverse=True)[0]
    
    ## 根据分数进行排序，得到元组列表，第一个元组中包含关系名和分数
    relation_rank = sorted(rel_scores.items(), key=lambda item:item[1], reverse=True)
#     print("---------- relation rank below ----------")
#     print("relations: " + str(relation_rank))
#     print("---------- relation rank above ----------")
    return relation_rank[0][0]
        
        
def get_answer(keywords, triples):
    if(keywords == None or triples == None):
        return None
    
    mainEntity = ""
    if(len(triples) >= 1):
        mainEntity = myquery.mid2name(triples[0][0])
    
    tmp = keywords.copy()
    for key in tmp:
        if(mainEntity != "" and key in mainEntity and len(keywords) > 1):
            keywords.remove(key)
        
    time1 = time.time()
    relation = get_top_relation(keywords, [x[1] for x in triples])
    time2 = time.time()
    print("-------- top relation below --------")
    print("top relation: " + str(relation))
    print("get top relation time used: " + str(time2-time1) + "s")
    print("-------- top relation above --------\n")
    answers = [x for x in triples if x[1] == relation]
    
    for ans in answers:
        if(ans[3] == 1):
            return ans
        
    return answers[0]

@app.route('/')
@app.route('/index')
def index():
    bgid = random.randint(1, 146)
    return render_template('index.html', num=bgid)

@app.route('/', methods=['post'])
def submit_form():
    ## 参数设置部分
    answerNum = 10
    score_threshold = 0
    show_threshold = 30
    
    ## 回答问题部分
    question = request.form['question']
    topEntities = QA.get_results(question)
    topEntities = topEntities[topEntities['predict'] > score_threshold].head(answerNum)
    keywords = QA.ret_keywords()

    ## 候选答案处理部分
    answerIds = topEntities['topic_words']
    answerNames = topEntities['topic_words_names']
    result = {"mid":answerIds.tolist(), "name":answerNames.tolist(), "type":[], "qid":[], "wid":[], "wikititle":[], "url":[]}
    print("------ subject candidates below ------")
    print(answerNames.tolist())
    print("------ subject candidates above -------\n")
    
    
    for mid in answerIds:
        result['type'].append(myquery.mid2type(mid))
        qid = myquery.mid2qid(mid)
        result['qid'].append(qid)
        if(qid != None):
            wid = myquery.qid2wid_en(qid)
        else:
            wid = myquery.mid2wid(mid)
        result['wid'].append(wid)
        if(wid != None):
            result['url'].append("https://en.wikipedia.org/wiki?curid=%s" % wid)
        elif(qid != None):
            if(qid.startswith('Q')): myurl = "https://www.wikidata.org/wiki/" + str(qid) 
            else: myurl = "https://www.wikidata.org/wiki/Property:" + str(wid)
            result['url'].append(myurl)
        else:
            result['url'].append(None)
        result['wikititle'].append(myquery.mid2wikititle(mid))

    data = pd.DataFrame(result).to_dict(orient="records")
    
    ## 处理主实体，并根据其关系获取答案
    ## TODO: 这里默认是一定能够找到主实体，处理找不到的情况
    time1 = time.time()
    mainEntity = {}
    mainEntity['title'] = data[0]['wikititle']
    if(mainEntity['title'] ==  None):
        mainEntity['title'] = data[0]['name']
    mainEntity['url'] = data[0]['url']
    mainEntity['type'] = data[0]['type']
    (mainEntity['image'], mainEntity['summary'], graphData, triples) = get_mainentity_info(data[0]['mid'], show_threshold)
    time2 = time.time()
    
    print("------ main entity below ------")
    print("mainEntity-name: " + str(mainEntity['title']))
    print("mainEntity-type: " + str(mainEntity['type']))
    print("get mainEntity time used: " + str(time2-time1) + "s")
    print("------ main entity above -------\n")
    
    ## 根据三元组获取答案部分
    ans = get_answer(keywords, triples)
    answer = {'name':None, 'url':None}
#     ## 这是从 wikidata获取数据的情况
#     if(ans != None):
#         print(ans)
#         if(ans[3] == 1):
#             qid = ans[2].split(":")[0]
#             answer['name'] = myquery.qid2wikititle(qid)
#             if(answer['name'] == None):
#                 answer['name'] = myquery.qid2name(qid)
#                 if(qid.startswith('Q')): myurl = "https://www.wikidata.org/wiki/" + str(qid) 
#                 else: myurl = "https://www.wikidata.org/wiki/Property:" + str(qid)
#                 answer['url'] = myurl
#             else:
#                 answer['url'] = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(answer['name'])
#         else:
#             answer['name'] = ans[2]
#             answer['url'] = None
    if(ans != None):
        print("--------- answer triple below ---------")
        print("Subject: " + str(myquery.mid2name(ans[0])))
        print("Relation: " + str(ans[1]))
        print("Object: " + str(myquery.mid2name(ans[2])))
        print("--------- answer triple above ---------\n")
        
        answer['name'] = myquery.mid2name(ans[2])
        wikititle = myquery.mid2wikititle(ans[2])
        if(wikititle != None):
            answer['url'] = "https://en.wikipedia.org/wiki/" + wikititle.replace(" ","_")
        else:
            answer['url'] = "https://en.wikipedia.org/w/index.php?search=" + answer['name'].replace(" ","+")
    
    if(len(graphData) > 4*show_threshold):
        tmptriples = []
        tmplist = random.sample(range(len(graphData)//4), k=20)
        for index in tmplist:
            tmptriples.append(graphData[index*4])
            tmptriples.append(graphData[index*4+1])
            tmptriples.append(graphData[index*4+2])
            tmptriples.append(graphData[index*4+3])
        graphData = tmptriples
        
    graphData = str(graphData)
    graphData = Markup(graphData);
    
    otherEntities = data[1:] if(len(data) < answerNum) else data[1:answerNum]
    google_results = {}
    
    return render_template("results.html", question=question, answer=answer, data=graphData, main=mainEntity, other=otherEntities, google=google_results)

@app.route('/algorithm')
def show_algorithm():
    return render_template("algorithm.html")

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html')

if __name__ == '__main__':
    QA.load_data("../datas/zyt/FB2M_names.txt")
    app.run(host='0.0.0.0') 
