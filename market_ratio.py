import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from data_processing import CryptoDataFetcher

class CryptoDataProcessor:

    def __init__(self, fetcher):
        self.fetcher = fetcher
        self.df = pd.DataFrame()

    # скачивание цен за определённый период
    def download_prices(self, symbol, time_frame, limit):
        crypto_prices = self.fetcher.price(symbol, time_frame, limit)['close']
        new_column = pd.Series(crypto_prices, name=symbol)
        self.df = pd.concat([self.df, new_column], axis=1)

    # расчёт процентного изменения
    def calculate_pct_change(self):
        df_pct_change = self.df.pct_change() * 100
        df_pct_change['mean'] = df_pct_change.mean(axis=1)
        return df_pct_change.dropna()

    # расчёт отклонения
    def adjust_values(self, row):
        mean = row['mean']
        adjustment = mean if mean > 0 else - abs(mean)
        return row.values - adjustment

    # поиск выбросов
    def find_outliers(self, result_cumsum):
        row = result_cumsum.iloc[-1]
        Q1 = np.percentile(row, 30)
        Q3 = np.percentile(row, 70)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        pos_outliers = row[row > upper_bound]
        neg_outliers = row[row < lower_bound]

        return pos_outliers, neg_outliers

class CryptoAnalysis:

    def __init__(self, api_key, api_secret, symbols):
        self.fetcher = CryptoDataFetcher(api_key, api_secret)
        self.processor = CryptoDataProcessor(self.fetcher)
        self.symbols = symbols

    def run(self):
        for symbol in tqdm(self.symbols, desc="Обработка криптовалют"):
            self.processor.download_prices(symbol, '1h', 168)
            # self.processor.download_prices(symbol, '15m', 96)  # можно раскомментировать

        file_path = r'C:\Programs\Screeners\data\market_ratio\market_ratio.csv'
        self.processor.df.to_csv(file_path, index=False)
        print('Данные успешно скачаны!')

        # расчёт процентного изменения
        df_pct_change_cleaned = self.processor.calculate_pct_change()

         # расчёт отклонения
        adjusted_values = df_pct_change_cleaned.apply(self.processor.adjust_values, axis=1)
        result = pd.DataFrame(adjusted_values.tolist(), columns=df_pct_change_cleaned.columns)
        result_cumsum = result.cumsum()

        # поиск выбросов
        pos_outliers, neg_outliers = self.processor.find_outliers(result_cumsum)

        # вывод результатов
        print(f"\nПоложительные выбросы: {len(pos_outliers)}")
        print(pos_outliers.sort_values(ascending=False))

        print(f"\nОтрицательные выбросы: {len(neg_outliers)}")
        print(neg_outliers.sort_values(ascending=True))

# Пример использования:
api_key = ''
api_secret = ''
print('\nРаботает скринер Market Ratio...')
list_futures = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BCHUSDT',
                 'LTCUSDT', 'ADAUSDT', 'ETCUSDT', 'LINKUSDT', 'TRXUSDT',
                   'DOTUSDT', 'DOGEUSDT', 'SOLUSDT', 'MATICUSDT', 'BNBUSDT', 'UNIUSDT', 'ICPUSDT', 'AAVEUSDT', 'FILUSDT', 'XLMUSDT', 'ATOMUSDT', 'XTZUSDT', 'SUSHIUSDT', 'AXSUSDT', 'THETAUSDT', 'AVAXUSDT', 'MANAUSDT', 'SANDUSDT', 'DYDXUSDT', 'NEARUSDT', 'EGLDUSDT', 'KSMUSDT', 'ARUSDT', 'FTMUSDT', 'PEOPLEUSDT', 'LRCUSDT', 'NEOUSDT', 'ALGOUSDT', 'IOTAUSDT', 'ENJUSDT', 'GMTUSDT', 'ZILUSDT', 'IOSTUSDT', 'APEUSDT', 'RUNEUSDT', 'KNCUSDT', 'APTUSDT', 'CHZUSDT', 'ROSEUSDT', 'ZRXUSDT', 'KAVAUSDT', 'ENSUSDT', 'SXPUSDT', 'OPUSDT', 'RSRUSDT', 'SNXUSDT', 'STORJUSDT', 'COMPUSDT', 'IMXUSDT', 'FLOWUSDT', 'QTUMUSDT', 'MASKUSDT', 'WOOUSDT', 'GRTUSDT', 'BANDUSDT', 'STGUSDT', 'ONEUSDT', 'JASMYUSDT', 'MKRUSDT', 'BATUSDT', 'MAGICUSDT', 'LDOUSDT', 'BLURUSDT', 'MINAUSDT', 'CFXUSDT', 'ASTRUSDT', 'GMXUSDT', 'ANKRUSDT', 'ACHUSDT', 'FETUSDT', 'FXSUSDT', 'HOOKUSDT', 'BNXUSDT', 'SSVUSDT', 'LQTYUSDT', 'STXUSDT', 'TRUUSDT', 'HBARUSDT', 'INJUSDT', 'BELUSDT', 'COTIUSDT', 'VETUSDT', 'ARBUSDT', 'KLAYUSDT', 'FLMUSDT', 'OMGUSDT', 'CKBUSDT', 'IDUSDT', 'LITUSDT', 'JOEUSDT', 'TLMUSDT', 'HOTUSDT', 'CHRUSDT', 'RDNTUSDT', 'ICXUSDT', 'ONTUSDT', 'UNFIUSDT', 'NKNUSDT', 'ARPAUSDT', 'DARUSDT', 'SFPUSDT', 'SKLUSDT', 'RVNUSDT', 'CELRUSDT', 'SPELLUSDT', 'SUIUSDT', 'IOTXUSDT', 'STMXUSDT', 'BSVUSDT', 'TONUSDT', 'GTCUSDT', 'DENTUSDT', 'ORDIUSDT', 'KEYUSDT', 'LEVERUSDT', 'QNTUSDT', 'MAVUSDT', 'XVGUSDT', 'AGLDUSDT', 'WLDUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'YGGUSDT', 'OGNUSDT', 'LPTUSDT', 'BNTUSDT', 'BAKEUSDT', 'LOOMUSDT', 'BIGTIMEUSDT', 'ORBSUSDT', 'WAXPUSDT', 'POLYXUSDT', 'TIAUSDT', 'MEMEUSDT', 'PYTHUSDT', 'JTOUSDT', 'ACEUSDT', 'XAIUSDT', 'MANTAUSDT', 'ALTUSDT', 'JUPUSDT', 'ZETAUSDT', 'STRKUSDT', 'PIXELUSDT', 'DYMUSDT', 'WIFUSDT', 'AXLUSDT', 'BOMEUSDT', 'METISUSDT', 'NFPUSDT', 'VANRYUSDT', 'AEVOUSDT', 'ETHFIUSDT', 'OMUSDT', 'ONDOUSDT', 'CAKEUSDT', 'PORTALUSDT', 'NTRNUSDT', 'KASUSDT', 'AIUSDT', 'ENAUSDT', 'WUSDT', 'TNSRUSDT', 'SAGAUSDT', 'TAOUSDT', 'FRONTUSDT', 'ATAUSDT', 'SUPERUSDT', 'ONGUSDT', 'LSKUSDT', 'GLMUSDT', 'REZUSDT', 'XVSUSDT', 'MOVRUSDT', 'BBUSDT', 'NOTUSDT', 'BICOUSDT', 'HIFIUSDT']

analysis = CryptoAnalysis(api_key, api_secret, list_futures)
analysis.run()