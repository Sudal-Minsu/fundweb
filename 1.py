from functions import check_account, get_auth_info,get_api_keys,get_access_token

app_key, app_secret = get_api_keys()
get_access_token(app_key, app_secret)
