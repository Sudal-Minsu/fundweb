import json
import keyring

# MySQL 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "0000",
    "database": "news_db",
    "port": 3306,
    "charset": "utf8mb4"
}


keyring.set_password('mock_app_key', '진상원', 'PS7xq8LmYdc2l9Olkr4Tyvpk1dtQAtMlNaS6')
keyring.set_password('mock_app_secret', '진상원', 'dBC6nh6uLNIiqW5EAi3uKXtLsjm8+rqoKXOrSt8VZWsewcFg4n/zFjpC0roThrNHEhI1vltKATr94Hxbl4UxHULF0lXdgiLtdG4aewQwa8irW4FET9D5Q00q83ohZqpeRrEk7VFlGL2vGQztiOtsfGhraewpy6RFubDS5sX7ilh1rDufXNI=')


def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '진상원')
    app_secret = keyring.get_password('mock_app_secret', '진상원')
    return app_key, app_secret

# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50127965",  # 계좌번호 앞 8자리
    "ACNT_PRDT_CD": "01"  # 계좌번호 뒤 2자리
}