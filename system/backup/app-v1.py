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
    graph = myquery.qid2triples(qid)
    graphdata = []
    for tup in graph:
        for item in tup:
            graphdata.append(item)
    summary = myquery.getSummaryById(wid)
    return image,summary,graphdata

@app.route('/')
@app.route('/index')
def index():
    bgid = random.randint(1, 146)
    return render_template('index.html', num=bgid)

@app.route('/', methods=['post'])
def submit_form():
    answerNum = 10
    question = request.form['question']
    
    topEntities = QA.get_results(question)
    topEntities = topEntities.head(answerNum)
    
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
    (mainEntity['image'], mainEntity['summary'], graphData) = get_mainentity_info(data[0]['mid'])
    
    tmptriples = []
    if(len(graphData) > 80):
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
    
#     time1 = time.time()
#     google_results = get_google_results(question, 5)
#     time2 = time.time()
#     print("Time used: "+str(time2-time1) + "s")
    google_results = {}
    
    return render_template("results.html", question=question, data=graphData, main=mainEntity, other=otherEntities, google=google_results)

@app.route('/algorithm')
def show_algorithm():
    return render_template("algorithm.html")

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html')

if __name__ == '__main__':
    QA.load_data("../datas/zyt/FB2M_names.txt")
    app.run(host='0.0.0.0') 
