import numpy as np
import pandas as pd

from data_processing import find_anomalies, futures_volume

list_futures = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BCHUSDT',
                 'LTCUSDT', 'ADAUSDT', 'ETCUSDT', 'LINKUSDT', 'TRXUSDT',
                   'DOTUSDT', 'DOGEUSDT', 'SOLUSDT', 'MATICUSDT', 'BNBUSDT', 'UNIUSDT', 'ICPUSDT', 'AAVEUSDT', 'FILUSDT', 'XLMUSDT', 'ATOMUSDT', 'XTZUSDT', 'SUSHIUSDT', 'AXSUSDT', 'THETAUSDT', 'AVAXUSDT', 'MANAUSDT', 'SANDUSDT', 'DYDXUSDT', 'NEARUSDT', 'EGLDUSDT', 'KSMUSDT', 'ARUSDT', 'FTMUSDT', 'PEOPLEUSDT', 'LRCUSDT', 'NEOUSDT', 'ALGOUSDT', 'IOTAUSDT', 'ENJUSDT', 'GMTUSDT', 'ZILUSDT', 'IOSTUSDT', 'APEUSDT', 'RUNEUSDT', 'KNCUSDT', 'APTUSDT', 'CHZUSDT', 'ROSEUSDT', 'ZRXUSDT', 'KAVAUSDT', 'ENSUSDT', 'SXPUSDT', 'OPUSDT', 'RSRUSDT', 'SNXUSDT', 'STORJUSDT', 'COMPUSDT', 'IMXUSDT', 'FLOWUSDT', 'QTUMUSDT', 'MASKUSDT', 'WOOUSDT', 'GRTUSDT', 'BANDUSDT', 'STGUSDT', 'ONEUSDT', 'JASMYUSDT', 'MKRUSDT', 'BATUSDT', 'MAGICUSDT', 'LDOUSDT', 'BLURUSDT', 'MINAUSDT', 'CFXUSDT', 'ASTRUSDT', 'GMXUSDT', 'ANKRUSDT', 'ACHUSDT', 'FETUSDT', 'FXSUSDT', 'HOOKUSDT', 'BNXUSDT', 'SSVUSDT', 'LQTYUSDT', 'STXUSDT', 'TRUUSDT', 'HBARUSDT', 'INJUSDT', 'BELUSDT', 'COTIUSDT', 'VETUSDT', 'ARBUSDT', 'KLAYUSDT', 'FLMUSDT', 'OMGUSDT', 'CKBUSDT', 'IDUSDT', 'LITUSDT', 'JOEUSDT', 'TLMUSDT', 'HOTUSDT', 'CHRUSDT', 'RDNTUSDT', 'ICXUSDT', 'ONTUSDT', 'UNFIUSDT', 'NKNUSDT', 'ARPAUSDT', 'DARUSDT', 'SFPUSDT', 'SKLUSDT', 'RVNUSDT', 'CELRUSDT', 'SPELLUSDT', 'SUIUSDT', 'IOTXUSDT', 'STMXUSDT', 'BSVUSDT', 'TONUSDT', 'GTCUSDT', 'DENTUSDT', 'ORDIUSDT', 'KEYUSDT', 'LEVERUSDT', 'QNTUSDT', 'MAVUSDT', 'XVGUSDT', 'AGLDUSDT', 'WLDUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'YGGUSDT', 'OGNUSDT', 'LPTUSDT', 'BNTUSDT', 'BAKEUSDT', 'LOOMUSDT', 'BIGTIMEUSDT', 'ORBSUSDT', 'WAXPUSDT', 'POLYXUSDT', 'TIAUSDT', 'MEMEUSDT', 'PYTHUSDT', 'JTOUSDT', 'ACEUSDT', 'XAIUSDT', 'MANTAUSDT', 'ALTUSDT', 'JUPUSDT', 'ZETAUSDT', 'STRKUSDT', 'PIXELUSDT', 'DYMUSDT', 'WIFUSDT', 'AXLUSDT', 'BOMEUSDT', 'METISUSDT', 'NFPUSDT', 'VANRYUSDT', 'AEVOUSDT', 'ETHFIUSDT', 'OMUSDT', 'ONDOUSDT', 'CAKEUSDT', 'PORTALUSDT', 'NTRNUSDT', 'KASUSDT', 'AIUSDT', 'ENAUSDT', 'WUSDT', 'TNSRUSDT', 'SAGAUSDT', 'TAOUSDT', 'FRONTUSDT', 'ATAUSDT', 'SUPERUSDT', 'ONGUSDT', 'LSKUSDT', 'GLMUSDT', 'REZUSDT', 'XVSUSDT', 'MOVRUSDT', 'BBUSDT', 'NOTUSDT', 'BICOUSDT', 'HIFIUSDT']
print('\nРаботает скринер Isolation Forest...')

# скачивание и сохранение данных
volumes, prices = futures_volume(list_futures, '1h', 168)
result = find_anomalies(volumes.T, cont=0.1)
futures = result[result['cluster'] != 1].index

print(f'\nФьючерсы с аномалиями:')
print(futures)