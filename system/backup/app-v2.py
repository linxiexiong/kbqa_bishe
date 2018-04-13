from flask import *
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

def get_mainentity_info(mid):
    qid = myquery.mid2qid(mid)
    name = myquery.mid2name(mid)
    wid = myquery.qid2wid_en(qid)
    image = myquery.getImageById(wid)
    triples = myquery.qid2triples(qid)
    if(triples == None):
        triples = []
    graphdata = []
    for tup in triples:
        for item in tup:
            graphdata.append(item)
    summary = myquery.getSummaryById(wid)
    return image,summary,graphdata,triples


def word_similarity(string1, string2):
    list1 = word_tokenize(string1)
    list2 = word_tokenize(string2)
    
    count = 0
    for word in list1:
        if(word in list2):
            count = count + 1
    return count / len(list2)    

def word_similarity_with_list(string1, list2):
    list1 = word_tokenize(string1)    
    count = 0
    for word in list1:
        if(word in list2):
            count = count + 1
    return count / len(list2)

def get_top_relation(keywords, relations):
    if(relations == None or keywords == None):
        return None
    
    rel_set = set()
    for rel in relations:
        rel_set.add(rel)
    
    rel_scores = {}
    for rel in rel_set:
        ## 仅对英文名计算相似度
        word_list = word_tokenize(rel.split("#")[0])
        rel_scores[rel] = sorted([word_similarity_with_list(x, word_list) for x in keywords], reverse=True)[0]
    
    ## 根据分数进行排序，得到元组列表，第一个元组中包含关系名和分数
    return sorted(rel_scores.items(), key=lambda item:item[1], reverse=True)[0][0]
        
        
def get_answer(keywords, triples):
    if(keywords == None or triples == None):
        return None
    
    mainEntity = ""
    if(len(triples) >= 1):
        mainEntity = triples[0][0]
    
    tmp = keywords
    for key in tmp:
        if(mainEntity != "" and mainEntity in key):
            keywords.remove(key)
    
    relation = get_top_relation(keywords, [x[1] for x in triples])
    print(relation)
    
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
    answerNum = 10
    score_threshold = 0.005
    question = request.form['question']
    
    topEntities = QA.get_results(question)
    keywords = QA.ret_keywords()
    topEntities = topEntities.head(answerNum)
#     topEntities = topEntities[topEntities['predict'] > score_threshold].head(answerNum)

    answerIds = topEntities['topic_words']
    answerNames = topEntities['topic_words_names']
    result = {"mid":answerIds.tolist(), "name":answerNames.tolist(), "type":[], "qid":[], "wid":[], "wikititle":[], "url":[]}
    
    for mid in answerIds:
        result['type'].append(myquery.mid2type(mid))
        qid = myquery.mid2qid(mid)
        result['qid'].append(qid)
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
    
    mainEntity = {}
    mainEntity['title'] = data[0]['wikititle']
    if(mainEntity['title'] ==  None):
        mainEntity['title'] = data[0]['name']
    mainEntity['url'] = data[0]['url']
    mainEntity['type'] = data[0]['type']
    (mainEntity['image'], mainEntity['summary'], graphData, triples) = get_mainentity_info(data[0]['mid'])

    pprint(mainEntity)
    
    ans = get_answer(keywords, triples)
    answer = {'name':None, 'url':None}
    if(ans != None):
        print(ans)
        if(ans[3] == 1):
            qid = ans[2].split(":")[0]
            answer['name'] = myquery.qid2wikititle(qid)
            if(answer['name'] == None):
                answer['name'] = myquery.qid2name(qid)
                if(qid.startswith('Q')): myurl = "https://www.wikidata.org/wiki/" + str(qid) 
                else: myurl = "https://www.wikidata.org/wiki/Property:" + str(qid)
                answer['url'] = myurl
            else:
                answer['url'] = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(answer['name'])
        else:
            answer['name'] = ans[2]
            answer['url'] = None
    
    
    if(len(graphData) > 80):
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
