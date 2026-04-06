import requests
from bs4 import BeautifulSoup

DOC_URL = "https://yanglab.westlake.edu.cn/gsmap/document/software"


def search_docs(query: str):

    html = requests.get(DOC_URL).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    results = []

    for line in text.splitlines():
        if query.lower() in line.lower():
            results.append(line.strip())

    return "\n".join(results[:20])
