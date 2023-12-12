import json
import time


class WebSocketConnection:
    ticker = "btcusdt"
    timeframe = "miniTicker"
    data_entry = []
    # function to convert time to a much more understandable time
    def converttime(data):
        eventtime = time.localtime(data['E'] // 1000)
        eventtime = f"{eventtime.tm_hour}:{eventtime.tm_min}:{eventtime.tm_sec}"

    def on_message(connection, message):
        # Get data in json format since it is already in json format
        data = json.loads(message)["data"]
        eventtime = time.localtime(data['E'] // 1000)
        eventtime = f"{eventtime.tm_hour}:{eventtime.tm_min}:{eventtime.tm_sec}"

        print(eventtime, "\t", round(float(data['o']), 2), "\t",
              round(float(data['h']), 2), "\t", round(float(data['l']), 2),"\t",
              round(float(data['c']), 2))
        WebSocketConnection.data_entry.append(data)

class Binance:
    binance_uri = "wss://stream.binance.com:9443/stream?streams="

    def __init__(self, ticker, timeframe):
        self.ticker = ticker
        self.timeframe = timeframe

    def ticker_data(self):
        url = f"{self.binance_uri}{self.ticker}@{self.timeframe}"
        return url
