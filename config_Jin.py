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


keyring.set_password('mock_app_key', '진상원', 'PSPBOlZt804x1wymlTA9MlxDLXCqlCuOn6Fd')
keyring.set_password('mock_app_secret', '진상원', 'vOsDQtJnKV9FdnFLnpKSbOMaMNqR+N3kUM+ncfnKNxF9/Lv2c68ZwJoZIcfoXSnGWXFPC/+hOa1hAhFe9Jji+mLrM3GTdExEoOUejj3zN4vWuam2gVutBxcZoYvgKuP1NX1SEesbPkLng2ftdXZ8VHG1L3mBQj+XJl/qxjo+pEAz9dnnAdg=')


def get_api_keys():
    """저장된 API 키를 불러오는 함수"""
    app_key = keyring.get_password('mock_app_key', '진상원')
    app_secret = keyring.get_password('mock_app_secret', '진상원')
    return app_key, app_secret

# 계좌 정보 저장
ACCOUNT_INFO = {
    "CANO": "50157248",  # 계좌번호 앞 8자리
    "ACNT_PRDT_CD": "01"  # 계좌번호 뒤 2자리
}