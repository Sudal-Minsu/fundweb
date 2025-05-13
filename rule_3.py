from functions import get_api_keys, get_access_token, get_hashkey, fetch_and_save_trade, save_to_db, execute_order, get_auth_info
from config import ACCOUNT_INFO, DB_CONFIG
import requests
import json



app_key, app_secret, access_token = get_auth_info()

order_no = execute_order(
    stock_code="005930",
    quantity=10,
    order_type="매수",       
    order_style="시장가",    
    app_key=app_key,
    app_secret=app_secret,
    access_token=access_token
)

if order_no:
    fetch_and_save_trade(
        order_type="매수",
        order_no=order_no,
        access_token=access_token,
        app_key=app_key,
        app_secret=app_secret,
        db_config=DB_CONFIG,
        table_name="trade_breakout"
    )