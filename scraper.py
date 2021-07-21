import json
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os
import urllib.request
import time

from settings import SCRAPER_FOLDER

class YandexScraper:
    def __init__(self, folder=SCRAPER_FOLDER):
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.list_urls = []
        self.forder = folder

    def run(self, text):
        text = text.replace(' ', '%20')
        self.driver.get(f'https://yandex.ru/images/search?text={text}')

        value = 0
        for i in range(50): 
            self.driver.execute_script(f'scrollBy({value},+500);')
            value += 500
            print(i)
            time.sleep(5)
            self.download()

    def download(self):
        elements = self.driver.find_elements_by_xpath('//div[contains(@class,"serp-item serp-item_type_search")]')
        for element in elements[-20:]:
            data = element.get_attribute('data-bem')
            data = json.loads(data)
            try:
                img_href = data['serp-item']['img_href']
                if img_href != None and img_href not in self.list_urls:
                    self.list_urls.append(img_href)
                    format_file = img_href.split('.')[-1].lower()
                    if format_file not in ['jpg', 'png', 'jpeg']:
                        format_file = 'webp'
                    urllib.request.urlretrieve(img_href, os.path.join(self.forder,f'people_in_glases_{len(self.list_urls)}.{format_file}'))
            except Exception:
                pass


if __name__ == '__main__':
    scraper = YandexScraper()
    queries = ["человек в очках", "мужчина в очках", "девушка в очках", "корейцы в очках"]
    for query in queries:
        scraper.run(query)

