import pandas as pd
from binance.client import Client
from binance.enums import *
import time
api_key= "AH7Bh0AqJbQO34RxW0ooMu3BxRXZKOwtwNrprihorSqlBNDTHhH6TKotpytaNV9O"
api_secret= "legfmdMZNmxF0RT6DV7ROCNaFrmC7wikBrDEVCcX9WXDGkomLJzCtErYGVhvkCt1"

client = Client(api_key, api_secret)
interval = Client.KLINE_INTERVAL_1HOUR
start_date = "1 month ago UTC"
end_date = "now UTC"

klines = client.get_historical_klines("MATICUSDT", interval, "3000 days ago")
df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df.set_index('date', inplace=True)
df['close'] = df['close'].astype(float)
df['open'] = df['open'].astype(float)
df['change'] = (df['close'] / df['open'])
print(df)
df.to_csv('MATIC-USDT.csv')
