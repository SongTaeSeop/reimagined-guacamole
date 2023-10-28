import requests
import time
import re
from bs4 import BeautifulSoup
from write_csv import write_csv_list

def get_url_with_id(url, number):
    return url.format(number)

def pagination():
    return [get_url_with_id("https://www.biblei.com/openboard/bbs_list.php?board_id=prayer_b&page={}", i) for i in range(1,61)]

def get_response(url):
    response = requests.get(url)
    response.encoding = "euc-kr"
    return response

def get_id_list():
    id_list = []
    for url in pagination():
        time.sleep(1)
        response = get_response(url)
        if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                rows = soup.find_all("tr", "style1")
                for row in rows:
                    id = [link.get("href") for link in row.descendants if link.name == "a"]
                    id = id[0][19:25]
                    id_list.append(id)
                    
        else:
             print(response.status_code)
    return id_list[1:]

def get_contents(number):
    url = get_url_with_id("https://www.biblei.com/openboard/print_list.php?bbs_no=prayer_b&gul_no={}", number)
    response = get_response(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        result = soup.find_all("tr")[10]
        # 태그 제거
        result = result.text
        return result

id_list = get_id_list()
for id in id_list:
    time.sleep(5)
    result = get_contents(id)
    # \n, \r제거
    result = re.split("\n|\r", result)
    # 공백 제거
    result = [[i] for i in result if i != ""]

    # text 머리 추가
    result = [["text"], *result]
    write_csv_list(result, id+".csv")
