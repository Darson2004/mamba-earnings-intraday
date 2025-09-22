# pip install xbbg
from xbbg import blp
df = blp.bdib(
    ticker='AAPL US Equity',
    dt='2025-09-19',           # local exchange date
    eventType='TRADE',
    interval=1                 # minutes
)
# df has columns: open, high, low, close, volume, numEvents; index in local tz
# df.to_excel("AAPL_IGPV_2025-09-19.xlsx")
