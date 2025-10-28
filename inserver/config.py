import os
import json

SECRETS_PATH = os.path.expanduser("~/Quant/fundweb/secrets.json")

with open(SECRETS_PATH, "r") as f:
    secrets = json.load(f)

DB_CONFIG = secrets["DB_CONFIG"]
ACCOUNTS = secrets["accounts"]

def get_account(name: str):
    if name not in ACCOUNTS:
        raise ValueError(f"등록되지 않은 계정: {name}")
    return ACCOUNTS[name]

print(get_account("acc3"))