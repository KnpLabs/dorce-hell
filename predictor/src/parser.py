from lxml import html
import requests
from googletrans import Translator
import os
import time

DATA_ROOT = '/var/data/'
FILENAME = 'not-pron.txt'
XPATH = '//span[@class="lister-item-header"]/span[2]/a/text()'
translator = Translator()

for offset in range(0, 200):
    url = 'https://www.imdb.com/search/title/?genres=comedy&view=simple&start=' + str(offset*50) + '&explore=title_type,genres&ref_=adv_nxt'
    print("URL: ", url)
    page = requests.get(url)
    tree = html.fromstring(page.content)
    titleMovies = tree.xpath(XPATH)
    separator = '.'
    allMovies = separator.join(titleMovies)
    print('abc : ', separator.join(titleMovies))
    for titleMovie in translator.translate(allMovies, dest='fr').split('.'):
       with open(os.path.join(DATA_ROOT, FILENAME), 'a') as file:
            file.write(titleMovie.text + '\n')