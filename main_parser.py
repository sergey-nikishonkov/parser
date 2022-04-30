import requests as re
from bs4 import BeautifulSoup
from threading import Thread
from pandas import DataFrame, ExcelWriter
import os


def collect_data(url, headers, category, games):
    params = {
        'start': 0,
        'count': 15,
        'cc': 'RU',
        'l': 'russian',
        'v': 4,
    }
    params['category'] = category
    result = []
    total_count = re.get(url, headers=headers, params=params).json()['total_count']
    pagesize = 15

    def page_parser():
        response = re.get(url, headers=headers, params=params)
        soup = BeautifulSoup(response.json()['results_html'], 'html.parser')
        game_data = [
            (
                tag.find(class_="tab_item_name").text,
                tag['href'],
                tag.find(class_="discount_original_price").text if tag.find(class_="discount_original_price") else '',
                tag.find(class_="discount_pct").text if tag.find(class_="discount_pct") else '',
                tag.find(class_="discount_final_price").text if tag.find(class_="discount_final_price") else ''
            )
            for tag in soup.find_all(class_="tab_item")
        ]
        result.extend(game_data)
        params["start"] += pagesize

    pool = [Thread(target=page_parser) for _ in range(total_count // pagesize + 1)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()

    games[category] = result



def main():
    url = 'https://store.steampowered.com/contenthub/querypaginated/category/NewReleases/render/'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0',
        'Connection': 'keep - alive',
        'ec-ch-ua-platform': 'Windows',
        'Referer': 'https: // store.steampowered.com / category / action /',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 OPR/86.0.4363.23'
    }
    games = {}
    pool = [
        Thread(target=collect_data, kwargs={'url': url, 'headers': headers, 'category': category, 'games': games}) for
        category in ['action', 'adventure_and_casual', 'rpg', 'simulation', 'strategy', 'sports_and_racing']
    ]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()

    for key in games:
        df = DataFrame(
            {
                'Title': [i[0] for i in games[key]],
                'Link': [i[1] for i in games[key]],
                'Original price': [i[2] for i in games[key]],
                'Discount': [i[3] for i in games[key]],
                'Final price': [i[4] for i in games[key]],
            }
        )
        with ExcelWriter('./games.xlsx',
                         mode='a' if os.path.exists('./games.xlsx') else 'w',
                         if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=key)

if __name__ == '__main__':
    main()
