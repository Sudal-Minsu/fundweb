from functions import check_account, get_auth_info

app_key, app_secret, access_token = get_auth_info()
check_account(access_token, app_key, app_secret)
