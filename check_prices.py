#!/usr/bin/env python3
"""
記録したニュースの株価を後から調べる
"""

import json
import os
from datetime import datetime, timedelta
import yfinance as yf

LOG_FILE = "news_log.json"
RESULT_FILE = "price_results.json"

def load_json(filename, default):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    logs = load_json(LOG_FILE, [])
    results = load_json(RESULT_FILE, [])
    checked_ids = {r.get("timestamp") for r in results}

    for log in logs:
        if log["timestamp"] in checked_ids:
            continue

        ts = datetime.fromisoformat(log["timestamp"])
        # 1日以上経ったものだけ調べる
        if datetime.now() - ts < timedelta(days=1):
            continue

        entry = {
            "timestamp": log["timestamp"],
            "title": log["title"],
            "importance": log["importance"],
            "sentiment": log["sentiment"],
            "stocks": []
        }

        for stock in log.get("stocks", []):
            code = stock.get("code")
            if not code or not code.isdigit():
                continue
            ticker = f"{code}.T"
            try:
                data = yf.download(ticker, start=ts.date(), end=(ts + timedelta(days=5)).date(), progress=False)
                if not data.empty:
                    start_price = data["Close"].iloc[0]
                    end_price = data["Close"].iloc[-1]
                    change = (end_price - start_price) / start_price * 100
                    entry["stocks"].append({
                        "code": code,
                        "name": stock.get("name"),
                        "change_percent": round(change, 2)
                    })
            except Exception as e:
                print(f"株価取得失敗: {code} → {e}")

        if entry["stocks"]:
            results.append(entry)
            print(f"記録: {log['title'][:40]}...")

    save_json(RESULT_FILE, results)
    print(f"完了: {len(results)}件の結果を保存")

if __name__ == "__main__":
    main()
