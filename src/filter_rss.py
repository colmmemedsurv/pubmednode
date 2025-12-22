import os
import openai
import feedparser
import requests
from lxml import etree
from datetime import datetime

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
RSS_FEED_URL = (
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/"
    "1FKYAX__W2XmZZnH7wCJZ2gjg5p61zj0lAum4ErUZK11BzSsdZ/"
    "?limit=100"
)

OUTPUT_ACCEPTED = "output/filtered_feed.xml"
OUTPUT_REJECTED = "output/rejected_feed.xml"

openai.api_key = os.environ.get("OPENAI_API_KEY")

# --------------------------------------------------
# FETCH PUBMED RSS WITH HEADERS
# --------------------------------------------------
def fetch_pubmed_rss(url: str) -> feedparser.FeedParserDict:
    headers = {
        "User-Agent": "pubmednode/1.0 (contact: colmme.medsurv@gmail.com)",
        "Accept": "application/rss+xml, application/xml",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return feedparser.parse(response.text)

# --------------------------------------------------
# OPENAI CLASSIFICATION (USER-SUPPLIED QUERY)
# --------------------------------------------------
def is_head_and_neck_cancer(text: str) -> bool:
    prompt = f"""
You are a biomedical expert.
Answer ONLY "YES" or "NO".

Is the following paper related to head and neck cancer
(including oral, laryngeal, tonsil, oropharynx, pharyngeal, larynx,
hypopharynx, nasopharynx, nasal, thyroid, head and neck skin SCC,
salivary gland cancers, rare head and neck cancer)?

Paper:
{text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message["content"].strip().upper() == "YES"

# --------------------------------------------------
# RSS HELPERS
# --------------------------------------------------
def create_channel(title: str, link: str, description: str):
    rss = etree.Element("rss", version="2.0")
    channel = etree.SubElement(rss, "channel")

    etree.SubElement(channel, "title").text = title
    etree.SubElement(channel, "link").text = link
    etree.SubElement(channel, "description").text = description
    etree.SubElement(channel, "lastBuildDate").text = datetime.utcnow().isoformat()

    return rss, channel

def add_item(channel, entry):
    item = etree.SubElement(channel, "item")
    etree.SubElement(item, "title").text = entry.title
    etree.SubElement(item, "link").text = entry.link
    etree.SubElement(item, "guid").text = entry.id
    etree.SubElement(item, "description").text = entry.get("summary", "")
    etree.SubElement(item, "pubDate").text = entry.get("published", "")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    feed = fetch_pubmed_rss(RSS_FEED_URL)

    if not feed.entries:
        raise RuntimeError(
            "PubMed RSS returned zero entries. "
            "Check User-Agent and feed URL."
        )

    accepted_rss, accepted_channel = create_channel(
        "Filtered PubMed – Head and Neck Cancer",
        RSS_FEED_URL,
        "Papers classified as related to head and neck cancer"
    )

    rejected_rss, rejected_channel = create_channel(
        "Rejected PubMed Papers – Not Head and Neck Cancer",
        RSS_FEED_URL,
        "Papers rejected by the automated classifier"
    )

    accepted_count = 0
    rejected_count = 0

    for entry in feed.entries:
        text_blob = f"{entry.title}\n\n{entry.get('summary', '')}"

        try:
            if is_head_and_neck_cancer(text_blob):
                add_item(accepted_channel, entry)
                accepted_count += 1
            else:
                add_item(rejected_channel, entry)
                rejected_count += 1
        except Exception as e:
            print(f"Error processing entry: {e}")

    os.makedirs("output", exist_ok=True)

    etree.ElementTree(accepted_rss).write(
        OUTPUT_ACCEPTED,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8"
    )

    etree.ElementTree(rejected_rss).write(
        OUTPUT_REJECTED,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8"
    )

    print(f"Accepted papers: {accepted_count}")
    print(f"Rejected papers: {rejected_count}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
