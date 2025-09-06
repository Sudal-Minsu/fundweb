import json
import keyring

# MySQL 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "news_db",
    "port": 3306,
    "charset": "utf8mb4"
}


keyring.set_password('mock_app_key', '진상원', 'PSvNMEEXvFUo3DRIpE4L3bYOoV7JKDda3Y5Y')
keyring.set_password('mock_app_secret', '진상원', 'NB7Vh7GDYaIyAmOqO7xSLz/HapmFZ16XMG5+trpXH14d4j2BI1+56nC2Nde8kxTTB1QU1bHxnXOoryYzt/2X1bOmWj3I0EZvUhdJi1TvxUAN3YE5fSDhUDWatUvU8khlp9funqeysPsSTwnGTndYT1l0o+kPeQAlehp2qj+uCocSO/GfF5w=')


def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '진상원')
    app_secret = keyring.get_password('mock_app_secret', '진상원')
    return app_key, app_secret

# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50141972",  # 계좌번호 앞 8자리
    "ACNT_PRDT_CD": "01"  # 계좌번호 뒤 2자리
}