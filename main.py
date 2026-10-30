#!/usr/bin/env python3
"""
日本株ニュース → AI評価 → Discord通知
"""

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import feedparser
import requests
from openai import OpenAI

# ==============================
# 設定
# ==============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

MODEL = "gpt-4o-mini"

RSS_FEEDS = [
    {"name": "日経速報", "url": "https://assets.wor.jp/rss/rdf/nikkei/news.rdf"},
    {"name": "日経マーケット", "url": "https://assets.wor.jp/rss/rdf/nikkei/markets.rdf"},
    {"name": "Yahoo経済", "url": "https://news.yahoo.co.jp/rss/topics/business.xml"},
]

MAX_ARTICLES_PER_RUN = 10
MIN_IMPORTANCE = 3
SEEN_FILE = "seen_articles.json"

# ==============================
# ユーティリティ
# ==============================
def load_seen() -> set:
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_seen(seen: set):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen), f, ensure_ascii=False, indent=2)

def article_id(entry) -> str:
    raw = entry.get("id") or entry.get("link") or entry.get("title", "")
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def fetch_recent_articles(feeds: List[Dict], max_items: int = 30) -> List[Dict]:
    articles = []
    seen_ids = set()

    for feed_info in feeds:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:15]:
                aid = article_id(entry)
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                published = entry.get("published_parsed") or entry.get("updated_parsed")
                pub_dt = None
                if published:
                    pub_dt = datetime(*published[:6], tzinfo=timezone.utc)

                articles.append({
                    "id": aid,
                    "title": entry.get("title", "").strip(),
                    "summary": entry.get("summary", entry.get("description", "")).strip()[:800],
                    "link": entry.get("link", ""),
                    "source": feed_info["name"],
                    "published": pub_dt.isoformat() if pub_dt else None,
                })
        except Exception as e:
            print(f"[WARN] RSS取得失敗: {feed_info['name']} → {e}")

    articles.sort(key=lambda x: x["published"] or "", reverse=True)
    return articles[:max_items]

# ==============================
# AI評価
# ==============================
SYSTEM_PROMPT = """あなたは日本株専門の超正確な証券アナリストAIです。
与えられたニュースを分析し、東証上場の日本企業に本当に関係がある情報だけを抽出してください。

【絶対に守るルール】
1. 東証（プライム・スタンダード・グロース）に上場している日本企業以外の銘柄は絶対に出力しない
2. 証券コードは必ず正確な4桁の数字で出す（例: 7203）。曖昧な場合は出力しない
3. 会社名は正式名称に近い形で書く
4. 関連度が「高」と言える明確な理由がない場合は、その銘柄を出さない
5. 単なる相場全体の話、海外企業中心、曖昧な業界話だけの記事は「関連なし」と判断する
6. 出力は指定のJSON形式のみ。余計な説明文は一切書かない

【重要度の基準】
5: 業績上方/下方修正、大型M&A、巨額受注、上場廃止関連など株価に直結する重大ニュース
4: 決算発表（好悪が明確）、増資・自社株買い、大型提携
3: 具体的な受注・新製品・規制関連など中程度の影響
2: 間接的・業界全体の話
1: ほとんど影響なし

【関連度の基準】
- 高: その会社が記事の主役、または直接的に大きな影響を受ける
- 中: 名前は出るが影響は限定的
- 低: ほとんど関係ない → 出力しない

【出力するJSON形式】
{
  "should_notify": trueまたはfalse,
  "importance": 1から5の整数,
  "sentiment": "ポジティブ" または "ネガティブ" または "中立",
  "summary": "2〜3文の簡潔で正確な要約",
  "related_stocks": [
    {
      "code": "7203",
      "name": "トヨタ自動車",
      "relevance": "高",
      "reason": "具体的な理由を1文で"
    }
  ],
  "reason_for_notify": "通知する/しない理由を簡潔に1文で"
}

- importanceが3以上、かつrelated_stocksに「高」が1つ以上ある場合のみ should_notify を true にする
- related_stocksは最大3銘柄まで。関連がなければ空の配列 [] にする
- 証券コードに自信がない場合は、その銘柄を絶対に出さない
"""

def analyze_article(client: OpenAI, article: Dict) -> Optional[Dict]:
    user_content = f"""以下のニュース記事を分析してください。

タイトル: {article['title']}
ソース: {article['source']}
要約/本文: {article['summary']}
リンク: {article['link']}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"[ERROR] AI分析失敗: {article['title'][:40]}... → {e}")
        return None

# ==============================
# Discord通知
# ==============================
def send_discord(article: Dict, analysis: Dict):
    if not DISCORD_WEBHOOK_URL:
        print("[SKIP] Discord Webhookが未設定です")
        return

    sentiment = analysis.get("sentiment", "中立")
    color_map = {
        "ポジティブ": 0x00C853,
        "ネガティブ": 0xD32F2F,
        "中立": 0x757575,
    }
    color = color_map.get(sentiment, 0x757575)

    stocks_text = ""
    for s in analysis.get("related_stocks", []):
        stocks_text += f"**{s.get('code')} {s.get('name')}**（{s.get('relevance')}）\n→ {s.get('reason')}\n"

    if not stocks_text:
        stocks_text = "関連銘柄なし"

    embed = {
        "title": article["title"][:250],
        "url": article["link"],
        "description": analysis.get("summary", "")[:1500],
        "color": color,
        "fields": [
            {"name": "重要度", "value": f"{analysis.get('importance', '?')}/5", "inline": True},
            {"name": "センチメント", "value": sentiment, "inline": True},
            {"name": "ソース", "value": article["source"], "inline": True},
            {"name": "関連銘柄", "value": stocks_text[:1000] or "なし", "inline": False},
            {"name": "通知理由", "value": analysis.get("reason_for_notify", "")[:500], "inline": False}
        ],
        "footer": {"text": f"日本株ニュースBot • {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
    }

    payload = {
        "username": "日本株ニュースBot",
        "embeds": [embed]
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"[OK] Discord通知送信: {article['title'][:40]}...")
    except Exception as e:
        print(f"[ERROR] Discord送信失敗: {e}")

# ==============================
# メイン処理
# ==============================
def main():
    print("=" * 50)
    print("日本株RSS → AI評価 → Discord")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    if not OPENAI_API_KEY:
        print("[エラー] OPENAI_API_KEY が設定されていません")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    seen = load_seen()

    print("\n[1] RSS取得中...")
    articles = fetch_recent_articles(RSS_FEEDS, max_items=MAX_ARTICLES_PER_RUN * 2)
    print(f"取得記事数: {len(articles)}")

    new_articles = [a for a in articles if a["id"] not in seen]
    print(f"未処理記事数: {len(new_articles)}")

    if not new_articles:
        print("新着なし。終了します。")
        return

    processed = 0
    for article in new_articles[:MAX_ARTICLES_PER_RUN]:
        print(f"\n--- 処理中: {article['title'][:50]}...")
        analysis = analyze_article(client, article)

        if analysis is None:
            continue

        seen.add(article["id"])
        processed += 1

        print(f"  重要度: {analysis.get('importance')} / センチメント: {analysis.get('sentiment')}")
        print(f"  通知判定: {analysis.get('should_notify')}")

        if analysis.get("should_notify") and analysis.get("importance", 0) >= MIN_IMPORTANCE:
            send_discord(article, analysis)
        else:
            print("  → 通知スキップ（条件未満）")

        time.sleep(1.2)

    save_seen(seen)
    print(f"\n処理完了: {processed}件")
    print("=" * 50)

if __name__ == "__main__":
    main()
