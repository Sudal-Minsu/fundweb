import keyring

# MySQL
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "port": 3306,
    "database": "news_db",
    "charset": "utf8mb4"
}

keyring.set_password('mock_app_key', '최진혁', 'PSbWOQW9CsjVIq8MwF3oeHG9gY9JjLHJVu8t')
keyring.set_password('mock_app_secret', '최진혁', 'uzxSVMytr/jWcbCYMBGcRMloeCM9A1fiTOur3Y3j30RY6gtvf3G0Bn1y/z6J2pa0CKKZRFf6OXpk/umYfxZaWQr4eVmoCJG6BX7wfQ/GOYlEDotyouzkMwevv7hjI06tzruSpPuN6EMS1nirtIeTnh8kxxN4LBS70XggdFevyM3KR87RG7k=')

def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '최진혁')
    app_secret = keyring.get_password('mock_app_secret', '최진혁')
    return app_key, app_secret

# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50139282",  # 계좌번호 앞 8자리
    "ACNT_PRDT_CD": "01"  # 계좌번호 뒤 2자리
}
