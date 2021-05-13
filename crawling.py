import selenium
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome(executable_path='chromedriver')
driver.implicitly_wait(3)

url = 'https://www.acmicpc.net/problem/tags'
print(url)
driver.get(url)
driver.implicitly_wait(1)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
html_tags = soup.select("a[href*='/problem/tag/']")
tags_title = []
tags_href = []
problem_set = []

for i in html_tags[1::2]:
    tags_title.append(i.get_text())
    tags_href.append(i.attrs['href'])

for i in tags_href:
    tag_url = 'https://www.acmicpc.net' + i
    driver.get(tag_url)
    driver.implicitly_wait(1)
    tag_html = driver.page_source
    tag_soup = BeautifulSoup(tag_html, 'html.parser')

    pages = tag_soup.select("a[href*='problemset?sort=ac_desc&algo=']")
    problems: list = []
    if len(pages) > 1:
        for j in pages:
            print('https://www.acmicpc.net' + j.attrs['href'])
            driver.get('https://www.acmicpc.net' + j.attrs['href'])
            driver.implicitly_wait(1)
            page_html = driver.page_source
            page_soup = BeautifulSoup(page_html, 'html.parser')
            _problems = page_soup.find_all('td', class_='list_problem_id')
            print(_problems)
            for k in _problems:
                problems.append(k)
    else:
        problems = tag_soup.find_all('td', class_='list_problem_id')

    print(problems)

    for j in problems:
        problem_url = 'https://www.acmicpc.net/problem/' + j.get_text()
        driver.get(problem_url)
        driver.implicitly_wait(1)
        problem_html = driver.page_source
        problem_soup = BeautifulSoup(problem_html, 'html.parser')

        problem_title = problem_soup.select('#problem_title')[0].get_text()
        problem_description = problem_soup.select('#problem_description')[0].get_text() if len(problem_soup.select('#problem_description')) != 0 else ''
        problem_input = problem_soup.select('#problem_input')[0].get_text() if len(problem_soup.select('#problem_input')) != 0 else ''
        problem_output = problem_soup.select('#problem_output')[0].get_text() if len(problem_soup.select('#problem_output')) != 0 else ''
        problem_set.append([tags_title[tags_href.index(i)], j.get_text(), problem_title, problem_description, problem_input, problem_output])


problem_df = pd.DataFrame(problem_set, columns=['tag', 'id', 'title', 'description', 'input', 'output'])
problem_df.to_csv(f'problem1_df.csv', mode='w', encoding='utf-8-sig', header=True, index=False)

driver.close()
