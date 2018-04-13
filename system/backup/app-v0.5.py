from flask import *
import random
import json
import urllib.parse
import urllib.request
import wikiparser as parser
import wikipedia as wiki


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
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
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

@app.route('/')
@app.route('/index')
def index():
    bgid = random.randint(1, 146)
    return render_template('index.html', num=bgid)


@app.route('/', methods=['post'])
def submit_form():
    otherEntityNum = 5
    question = request.form['question']
  
    print(question)
    mainEntity = {}
    otherEntities = {}
    graphData = []

    answers = wiki.search(question)
    if(len(answers) >= 1):
        mainEntity['title'] = answers[0]
    else:
        mainEntity['title'] = wiki.random()

    mainEntity['url'] = get_url(mainEntity['title'])
    mainEntity['summary'] = get_summary(mainEntity['title'])
    mainEntity['image'] = parser.getMainImage(mainEntity['url'])['link']

    for i in range(1,len(answers)):
        if(i > otherEntityNum):
            break
        otherEntities[answers[i]] = get_url(answers[i])
        
    while(len(otherEntities) < otherEntityNum):
        tmpEnt = wiki.random()
        otherEntities[tmpEnt] = get_url(tmpEnt)

    google_results = google_results_clear(get_google_results(question, 5))

    return render_template("results.html", question=question, data=graphData, main=mainEntity, other=otherEntities, google=google_results)


@app.route('/algorithm')
def show_algorithm():
    return render_template("algorithm.html")


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
