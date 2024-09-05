from data_processing import CryptoDataFetcher, AnomalyDetector, CoreSearcher, DataVisualizer

# настройки
api_key = ''
api_secret = ''
fetcher = CryptoDataFetcher(api_key, api_secret)

# Скачивание данных
df = fetcher.price('BTCUSDT', '15m', 500)
data = df[['close', 'volume']].iloc[:]

# Обнаружение аномалий
params = [0.05, 0.07]
anomaly_detector = AnomalyDetector(params)
df, anomaly_1, anomaly_2 = anomaly_detector.detect_anomalies(data)

# Поиск ядер
core_searcher = CoreSearcher()
result = core_searcher.search_cores(df)
print(result)

# Визуализация
visualizer = DataVisualizer()
visualizer.plot_close_with_anomalies(df, anomaly_1, anomaly_2, result)
