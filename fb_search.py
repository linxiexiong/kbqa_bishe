import json
import urllib



#api_key = open(".api_key").read()
#query = 'blue bottle'
#proxy="http://sulixin:YmE0NDMzOD@23.106.154.47:443"
#service_url = 'https://www.google.com/search?q=knowledge+graph+search+api&kponly&kgmid=/m/05swzs_'

import requests

proxies = {
  "http": "http://23.106.154.47:443",
  "https": "https://sulixin:YmE0NDMzOD@23.106.154.47:443",
}

r=requests.get("https://www.wikidata.org/wiki/Special:EntityData/Q42.json")
print r.text