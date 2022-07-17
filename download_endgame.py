import os.path

import requests
from bs4 import BeautifulSoup


def main():
    # ".rtbz"
    download('endgames/6-WDL/', ".rtbw", 'https://tablebase.lichess.ovh/tables/standard/6-wdl/')
    download('endgames/6-WDL/', ".rtbw", 'https://tablebase.lichess.ovh/tables/standard/3-4-5/')


def download(local_path, target_extension, url):
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    count = 0
    for link in soup.find_all('a'):
        href_relative = link.get('href')
        if href_relative.endswith(target_extension):
            local_full = local_path + href_relative
            href_full = url + href_relative
            if os.path.exists(local_full):
                print(f"Skipping: {href_full}")
                continue
            count += 1
            print(f"{count} downloading: {href_full} to {local_full}")
            response = requests.get(href_full)
            with open(local_full, 'wb') as file_result:
                file_result.write(response.content)
    print(f"Downloaded: {count} files to {local_path} from {url} with extension {target_extension}")


if __name__ == '__main__':
    main()
