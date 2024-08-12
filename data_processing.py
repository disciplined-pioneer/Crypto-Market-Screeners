import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime, timedelta
from binance.um_futures import UMFutures

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

# сбор данных
class CryptoDataFetcher:

    def __init__(self, api_key: str, api_secret: str):
        self.client = UMFutures(key=api_key, secret=api_secret)

    @staticmethod
    def float_type(df: pd.DataFrame) -> pd.DataFrame:

        """Изменяет тип данных всех колонок DataFrame на float."""

        return df.astype(float)

    @staticmethod
    def last_bar_open(time_frame: str) -> int:

        current_time = datetime.now()
        first_bar_open_time = datetime(current_time.year, current_time.month, current_time.day, 3, 0, 0)

        bar_durations = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
        last_bar_open_time = {tf: first_bar_open_time + (
                (current_time - first_bar_open_time) // timedelta(minutes=bar_durations[tf])) * timedelta(
            minutes=bar_durations[tf]) for tf in bar_durations}

        return round(last_bar_open_time[time_frame].timestamp() * 1000)

    def price(self, futures: str, time_frame: str, limit: int) -> pd.DataFrame:

        """Получает информацию об исторических ценах криптовалюты."""

        inf = self.client.klines(futures, time_frame, limit=limit)
        rename_col = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                      "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

        df = pd.DataFrame(inf, columns=rename_col)
        df = df[["open", "high", "low", "close", "volume"]]
        df = self.float_type(df)

        return df

    def delta(self, futures: str, time_frame: str, limit: int) -> pd.DataFrame:

        """Получает информацию об исторической дельте криптовалюты."""

        taker_delta = self.client.taker_long_short_ratio(futures, time_frame, limit=limit)[1:]
        delta = pd.DataFrame(taker_delta)

        delta = self.float_type(delta)

        delta['delta'] = delta['buyVol'] - delta['sellVol']
        delta = delta[['buyVol', 'sellVol', 'delta']]

        if time_frame in ['1h', '4h', '1d']:
            st_time = self.last_bar_open(time_frame)
            info_bar = self.client.taker_long_short_ratio(futures, '5m', limit=500, startTime=st_time)[1:]
            bar = pd.DataFrame(info_bar)

            bar = self.float_type(bar)
            delta_bar = bar['buyVol'].sum() - bar['sellVol'].sum()
            delta.loc[len(delta)] = [bar['buyVol'].sum(), bar['sellVol'].sum(), delta_bar]
        else:
            delta = delta.shift(periods=-1)

        return delta.iloc[1:].reset_index(drop=True)

    def open_interest(self, futures: str, time_frame: str, limit: int) -> pd.DataFrame:

        """Получает информацию об историческом открытом интересе криптовалюты."""

        taker_OI = self.client.open_interest_hist(futures, time_frame, limit=limit)
        OI = pd.DataFrame(taker_OI)
        OI = OI.rename(columns={'sumOpenInterest': 'sumOI'})

        OI['Old'] = OI['sumOI'].shift(periods=1)
        OI = OI[['sumOI', 'Old']]

        OI = self.float_type(OI)

        OI['open interest'] = OI['sumOI'] - OI['Old']
        OI = OI[['sumOI', 'open interest']]
        OI = OI.dropna()

        if time_frame in ['1h', '4h', '1d']:
            st_time = self.last_bar_open(time_frame)
            info_bar = self.client.open_interest_hist(futures, '5m', limit=500, startTime=st_time)

            bar = pd.DataFrame(info_bar)
            bar = bar.rename(columns={'sumOpenInterest': 'sumOI'})

            bar['Old'] = bar['sumOI'].shift(periods=1)
            bar = bar[['sumOI', 'Old']]

            bar = self.float_type(bar)

            bar['open interest'] = bar['sumOI'] - bar['Old']
            OI_bar = bar[['sumOI', 'open interest']]

            OI_bar = OI_bar.dropna()
            OI.loc[len(OI)] = [OI_bar.iloc[-1]['sumOI'], OI_bar['open interest'].sum()]
        else:
            OI = OI.shift(periods=-1)

        return OI.iloc[1:].reset_index(drop=True)

    def download_info(self, futures: str, time_frame: str, limit: int) -> pd.DataFrame:

        """Скачивает и объединяет данные о ценах, дельте и открытом интересе."""

        dl = self.delta(futures, time_frame, limit)
        #open_interest = self.open_interest(futures, time_frame, limit)
        prices = self.price(futures, time_frame, limit).tail(len(dl)).reset_index(drop=True)

        df = pd.concat([prices, dl], axis=1)
        df = df[["open", "high", "low", "close", "volume", 'delta']]

        return df.dropna()

# скачивание данных об объёме фьючерсов (isol_forest)
def futures_volume(list_futures, time_frame, lim=100):
    data_vol = {}
    data_price = {}
    for fut in tqdm(list_futures, desc="Загрузка объемов фьючерсов"):
        df = fetcher.price(fut, time_frame=time_frame, limit=lim)
        data_vol[fut] = df['volume']
        data_price[fut] = df['close']
    return pd.concat(data_vol, axis=1), pd.concat(data_price, axis=1)

# поиск аномалий (isol_forest)
def find_anomalies(df, cont=0.2):
    pipe_islf = Pipeline([
        ('scaler', StandardScaler()),
        ('IsolationForest', IsolationForest(contamination=cont))
    ])
    df['cluster'] = pipe_islf.fit_predict(df)
    return df


# обнаружение аномалий в данных (volume_spikes)
class AnomalyDetector:

    def __init__(self, params):
        self.params = params

    def detect_anomalies(self, df):
        for param in self.params:
            df[f'anomaly_{param}'] = Pipeline([
                ('scaler', StandardScaler()),
                ('isolation_forest', IsolationForest(contamination=param))
            ]).fit_predict(df)

        # Обновление аномалий на основе условия
        df.loc[(df[f'anomaly_{self.params[0]}'] == -1) & (df[f'anomaly_{self.params[1]}'] == -1), f'anomaly_{self.params[1]}'] = 0

        # Кластеризация на основе наличия аномалий
        df['cluster'] = df.apply(lambda row: 0 if -1 in (row[f'anomaly_{self.params[0]}'], row[f'anomaly_{self.params[1]}']) else 1, axis=1)

        anomaly_1 = df[df[f'anomaly_{self.params[0]}'] == -1]
        anomaly_2 = df[df[f'anomaly_{self.params[1]}'] == -1]

        return df, anomaly_1, anomaly_2

# поиск ядер аномалий (volume_spikes)
class CoreSearcher:

    def __init__(self, step=10, threshold=5):
        self.step = step
        self.threshold = threshold

    def search_cores(self, df):
        arr = []
        for row in range(self.step, len(df)):
            data = df.iloc[row-self.step:row]
            count_ones = np.sum(data.cluster.values == 1)

            if count_ones <= self.threshold:
                arr.extend(list(data.index))

        result, start = [], arr[0]
        for i in range(1, len(arr)):
            if arr[i] - arr[i-1] > 1:
                result.append([start, arr[i-1]])
                start = arr[i]
        result.append([start, arr[-1]])

        return result

# визуализация данных и аномалий (volume_spikes)
class DataVisualizer:

    def __init__(self, figsize=(20, 10)):
        self.figsize = figsize

    def plot_close_with_anomalies(self, df, anomaly_1, anomaly_2, result):
        sns.set(style='darkgrid')
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)

        ax.plot(df['close'], color='red', label='Close Price')
        ax.plot(anomaly_1.index, anomaly_1['close'], 'bs', alpha=0.6, markersize=8, label='Глобальная аномалия')
        ax.plot(anomaly_2.index, anomaly_2['close'], 'ms', alpha=0.6, markersize=8, label='Локальная аномалия')

        ax.set_title('Закрытие и Аномалии')
        ax.set_xlabel('Индекс')
        ax.set_ylabel('Цена Закрытия')
        ax.legend()

        for start, end in result:
            data = df['close'].iloc[start:end]
            x = np.linspace(start, end, 100)
            plt.fill_between(x, data.min(), data.max(), color='blue', alpha=0.1)

        plt.tight_layout()
        plt.show()


# анализ сантимента (sentiment)
class SentimentAnalyzer:
    
    def __init__(self, api_key, api_secret, futures_list, time_frame='1h', limit=720):
        self.fetcher = CryptoDataFetcher(api_key, api_secret)
        self.futures_list = futures_list
        self.time_frame = time_frame
        self.limit = limit
        self.data_dir = r'C:\Programs\Screeners\data\Sentiment'
        self.sentiment_day = 0
        self.sentiment_week = 0
        self.sentiment_month = 0

    def fetch_and_save_data(self):
        for future in tqdm(self.futures_list, desc="Загрузка цен фьючерсов"):
            df = self.fetcher.price(future, self.time_frame, self.limit)
            file_path = os.path.join(self.data_dir, f'sentiment_{future}.csv')
            df.to_csv(file_path, index=False)
        print('Данные полностью скачаны!')

    def analyze_sentiment(self):
        for future in self.futures_list:
            file_path = os.path.join(self.data_dir, f'sentiment_{future}.csv')
            df = pd.read_csv(file_path)
            self._update_sentiment(df)

        self._print_results()

    def _update_sentiment(self, df):
        self.sentiment_day += self._calculate_sentiment(df, 0, 23)
        self.sentiment_week += self._calculate_sentiment(df, 0, 167)
        self.sentiment_month += self._calculate_sentiment(df, 0, -1)

    def _calculate_sentiment(self, df, start_idx, end_idx):
        price_old = df['open'].iloc[start_idx]
        price_now = df['close'].iloc[end_idx]
        return 1 if price_now > price_old else -1 if price_now < price_old else 0

    def _print_results(self):
        count = len(self.futures_list)
        self._print_sentiment_result('день', self.sentiment_day, count)
        self._print_sentiment_result('неделю', self.sentiment_week, count)
        self._print_sentiment_result('месяц', self.sentiment_month, count)

    def _print_sentiment_result(self, period, sentiment_value, count):
        sentiment_percentage = round(sentiment_value / count * 100, 2)
        print(f'\nРезультат работы за {period}: {sentiment_value}')
        print(f'Перевес составил: {sentiment_percentage}%\n')



# настройки
api_key = ''
api_secret = ''
fetcher = CryptoDataFetcher(api_key, api_secret)