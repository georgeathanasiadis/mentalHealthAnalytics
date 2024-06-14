import fitbit
import datetime
import pandas as pd
import gather_keys_oauth2 as Oauth2

CLIENT_ID = '23QQXR'
CLIENT_SECRET = '9e050c755faa79a60e8b8dc0d1e804ed'

def connect():
    server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
    server.browser_authorize()
    ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
    REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
    fitbit_api = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
    return fitbit_api
