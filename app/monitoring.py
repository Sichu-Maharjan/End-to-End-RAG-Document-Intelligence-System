import re
import os
import pandas as pd

LOG_PATH = "../processed/logs/app.log"

LINE = re.compile(
    r"^(?P<time>[\d\- :,]+) \| INFO \| question=(?P<question>.+?) \| "
    r"retrieval=(?P<retrieval>[\d.]+)s \| llm=(?P<llm>[\d.]+)s \| "
    r"total=(?P<total>[\d.]+)s \| sources=(?P<sources>\d+) \| idk=(?P<idk>\w+)"
)

def load_requests():
    """Parse question lines from the log into a table."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            m = LINE.match(line.strip())
            if m:
                d = m.groupdict()
                rows.append({
                    "time": d["time"],
                    "question": d["question"].strip("'\""),
                    "retrieval_s": float(d["retrieval"]),
                    "llm_s": float(d["llm"]),
                    "total_s": float(d["total"]),
                    "sources": int(d["sources"]),
                    "idk": d["idk"] == "True",
                })
    return pd.DataFrame(rows)

def count_by_level(level):
    if not os.path.exists(LOG_PATH):
        return 0
    with open(LOG_PATH) as f:
        return sum(1 for line in f if f" | {level} | " in line)