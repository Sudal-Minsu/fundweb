import json
import keyring

# MySQL 연결 설정
DB_CONFIG = {"host": "localhost", 
             "user": "stockuser", 
             "password": "stockpass123!", 
             "port": 3306, 
             "database": "news_db", 
             "charset": "utf8mb4", }

# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50150860",  # 계좌번호 앞 8지리
    "ACNT_PRDT_CD": "01",  # 계좌번호 뒤 2자리
}

#keyring.set_password('mock_app_key', '고민수', 'PSXtsebcvZLq1ZKGsppEYYxCd0RoOd48INlF')
#keyring.set_password('mock_app_secret', '고민수', 'pnPjHI+nULtuBz3jTzPhvBQY+9VKfMCql6lN3btyp19EGhi1hALeHrPjhsFj016eaGqACCcDWdZ3ivhNOIVhBZRATrHdiTk8L8uCxVNQn3qpWSk+54SQ/XMCyJvVpUSaPiRBf+n0iSu7blyUjBxQgt9zBMUvBB23ylyMg8yrWCDJZpgQXM4=')

def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '고민수')
    app_secret = keyring.get_password('mock_app_secret', '고민수')
    print("app_key:", app_key)
    print("app_secret:", app_secret)
    return app_key, app_secret
