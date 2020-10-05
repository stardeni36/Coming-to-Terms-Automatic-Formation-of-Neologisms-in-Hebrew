from bs4 import BeautifulSoup
from urllib.request import urlopen
import json
import time
from tqdm import tqdm

NUM_OF_PAGES = 30
JSON_PATH = "crawled_texts.json"

SLEEP_TIME = 0.2
BASE_URL = "https://benyehuda.org/read/"
PRINT_MODE = True

if __name__ == '__main__':

    heb_texts = {}

    cur_index = 1

    for i in tqdm(range(22000, 25000)):  # NUM_OF_PAGES):

        url = BASE_URL + str(i)
        try:
            source_data = urlopen(url).read()
            time.sleep(SLEEP_TIME)
        except:
            print(i, ": PAGE NOT FOUND")
            continue

        soup = BeautifulSoup(source_data, features="html.parser")
        heb_text = soup.find("div", {"class": "maintext-prose-body search-margin"})
        type = "prose"
        if not heb_text:
            heb_text = soup.find("div", {"class": "maintext-poetry-body search-margin"})
            type = "poetry"

        heb_texts[cur_index] = (heb_text.text, type, url)
        cur_index += 1

    with open("heb_texts7.json", 'w', encoding='utf8') as outfile:
        json.dump(heb_texts, outfile)