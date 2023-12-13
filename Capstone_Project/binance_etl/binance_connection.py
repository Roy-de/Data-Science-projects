class Binance:
    binance_uri = "wss://stream.binance.com:9443/stream?streams="

    def __init__(self, ticker, timeframe):
        self.ticker = ticker
        self.timeframe = timeframe

    def ticker_data(self):
        url = f"{self.binance_uri}{self.ticker}@{self.timeframe}"
        return url