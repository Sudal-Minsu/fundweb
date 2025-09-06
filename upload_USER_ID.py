import requests

SERVER_URL = "http://서버IP:5000"
USER_ID = "jinsw0129"
FILE_PATH = "rule_2_결과/full_kelly.csv"

with open(FILE_PATH, "rb") as f:
    res = requests.post(f"{SERVER_URL}/upload/{USER_ID}", files={"file": f})
    print(res.json())
