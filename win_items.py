from datetime import datetime

from urllib import request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from urllib.request import Request

import pandas as pd

import multiprocessing
import time

service = Service("C:\Instagram_stories_scrapper\chrome web driver\chromedriver_win32/chromedriver.exe")
drive = webdriver.Chrome(service=service)


def get_summoner_name():
    ladder_url = "https://euw.op.gg/ranking/ladder/"
    summoner = []
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    req = Request(ladder_url, headers=header)
    html = request.urlopen(req)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find_all("table", {"class": "ranking-table"})
    tbody = table[0].find("tbody")
    tr = tbody.find_all("tr", {"class": "ranking-table__row"})
    for i in range(0, 10):
        td = tr[i].find_all("td", {"class": "ranking-table__cell--summoner"})
        for summ in td:
            placeholder = summ.find_all('a')
            summoner.append(placeholder[0].get('href').split("=")[1])
    return summoner


def get_items(summoner_profile):
    champions = []
    results = []
    kda = []
    killdeathassist = []
    mode = []
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    req = Request(summoner_profile, headers=header)
    html = request.urlopen(req)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find_all("div", {"class": "GameItemWrap"})
    content = soup.find("div", {"class": "GameItemList"})
    for i in range(0, 10):
        champion = table[i].find("div", {"class": "ChampionName"}).get_text()
        game_type = table[i].find("div", {"class": "GameType"}).get_text()
        result = table[i].find("div", {"class": "GameResult"}).get_text()
        kdaRatio = table[i].find("span", {"class": "KDARatio"}).get_text()

        for k in range(len(table)):
            kda.append(kdaRatio)
        champions.append(champion.strip('\n'))
        results.append(result.strip('\n').strip('\t'))
        mode.append(game_type)  # .strip('\n').strip('\t'))
        killdeathassist.append(kdaRatio)
    return champions, killdeathassist, results, mode


def summoner_url(summoner):
    summoner_profile = []
    print(len(summoner))
    for i in range(len(summoner)):
        summoner_profile.append(("https://euw.op.gg/summoner/userName=" + summoner[i] + "/"))
    return summoner_profile


def get_item_name(summoner_profile):
    # item_number = ['item1','item2','item3','item4','item5','item6']
    build = []
    items = []
    chunk_size = 7
    for i in range(len(summoner_profile)):
        header = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        req = Request(summoner_profile, headers=header)
        html = request.urlopen(req)
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find("div", {"class": "GameItemList"})
        table = soup.find_all("div", {"class": "GameItemWrap"})
        for k in range(len(content)):
            for i in range(0, 10):

                item_list = table[i].find("div", {"class": "ItemList"})
                # item = item_list.find_all("div", {"class": "Item"})
                img = item_list.find_all('img', {"class": "Image tip"})
                for j in img:
                    try:
                        mystr = j['title']
                        start = mystr.find('>') + 1
                        end = mystr.find('<', 1)
                        items.append(mystr[start:end])
                    except KeyError:
                        pass
        build = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    return build


def get(lane, name):
    build = [
        ('starter_items_1', []),
        ('starter_items_2', []),
        ('build_1', []),
        ('build_2', []),
        ('build_3', []),
        ('build_4', []),
        ('build_5', []),
        ('boot_1', []),
        ('boot_2', []),
        ('boot_3', [])
    ]
    URL = "https://euw.op.gg/champion/" + name + "/statistics/" + lane
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    req = Request(URL, headers=header)
    html = request.urlopen(req)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find_all("table", {"class": "champion-overview__table"})
    tbody = table[1].find("tbody")

    tr = tbody.find_all("tr", {"class": "champion-overview__row"})

    for i in range(0, 10):
        td = tr[i].find("td", {"class": "champion-overview__data"})
        ul = td.find("ul", {"class": "champion-stats__list"})
        li = ul.find_all("li")
        for j in li:
            try:
                mystr = j['title']
                start = mystr.find('>') + 1
                end = mystr.find('<', 1)
                build[i][1].append(mystr[start:end])
            except KeyError:
                pass

    return build


def list_to_df(build, champions, killdeathassist, results, mode):
    df = pd.DataFrame(build, columns=['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7'])
    # print (df)
    df3 = pd.DataFrame(champions, columns=['champions'])
    df5 = pd.DataFrame(mode, columns=['mode'])
    df4 = pd.DataFrame(results, columns=['results'])
    df2 = pd.DataFrame(killdeathassist, columns=['kda_Ratio'])
    frames = [df3, df2, df4, df5, df]
    # columns=['champions','kill','death','assist','results','mode','item1','item2','item3','item4','item5','item6','item7']
    initial_d = pd.concat([df3, df2, df4, df5, df], axis=1, join='inner')
    return initial_d


def main(summoner_profile):
    for i in range(len(summoner_profile)):

        build = get_item_name(summoner_profile[i])
        champions, killdeathassist, results, mode = get_items(summoner_profile[i])
        dfs = list_to_df(build, champions, killdeathassist, results, mode)
        print(dfs)

        dfs.to_csv('C:/Users/Lenovo/Documents/ML/' + str(i) + datetime.now().strftime('%m-%d-%H-%M.csv'), index=False, header=True)
