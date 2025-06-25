from functions import check_account, get_auth_info

app_key, app_secret, access_token = get_auth_info()
res1, res2 = check_account(access_token, app_key, app_secret)
print(res1)