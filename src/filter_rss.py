import feedparser
import os
import openai
from lxml import etree
from datetime import datetime

RSS_FEED_URL = "https://pubmed.ncbi.nlm.nih.gov/rss/search/1FKYAX__W2XmZZnH7wCJZ2gjg5p61zj0lAum4ErUZK11BzSsdZ/?limit=100"
OUTPUT_FILE = "output/filtered_feed.xml"

openai.api_key = os.environ.get("OPENAI_API_KEY")

def is_head_and_neck_cancer(text: str) -> bool:
    prompt = f"""
You are a biomedical expert.
Answer ONLY "YES" or "NO".

Is the following paper related to head and neck cancer
(including oral, laryngeal, pharyngeal, nasal, salivary gland cancers)?

Paper:
{text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message["content"].strip().upper() == "YES"

def main():
    feed = feedparser.parse(RSS_FEED_URL)

    rss = etree.Element("rss", version="2.0")
    channel = etree.SubElement(rss, "channel")

    etree.SubElement(channel, "title").text = "Filtered PubMed – Head and Neck Cancer"
    etree.SubElement(channel, "link").text = RSS_FEED_URL
    etree.SubElement(channel, "description").text = (
        "Automatically filtered PubMed RSS feed for head and neck cancer"
    )
    etree.SubElement(channel, "lastBuildDate").text = datetime.utcnow().isoformat()

    for entry in feed.entries:
        text_blob = f"{entry.title}\n\n{entry.get('summary', '')}"

        try:
            if is_head_and_neck_cancer(text_blob):
                item = etree.SubElement(channel, "item")
                etree.SubElement(item, "title").text = entry.title
                etree.SubElement(item, "link").text = entry.link
                etree.SubElement(item, "guid").text = entry.id
                etree.SubElement(item, "description").text = entry.get("summary", "")
                etree.SubElement(item, "pubDate").text = entry.get("published", "")
        except Exception as e:
            print(f"Error processing entry: {e}")

    os.makedirs("output", exist_ok=True)
    etree.ElementTree(rss).write(
        OUTPUT_FILE,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8"
    )

if __name__ == "__main__":
    main()
