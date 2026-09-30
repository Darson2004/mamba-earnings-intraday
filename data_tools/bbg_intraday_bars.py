# pip install blpapi pytz  (or use Python 3.9+ zoneinfo)
import blpapi
from blpapi import SessionOptions, Session
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

def igpv_bars(ticker: str, trade_date: str, interval=1,
              local_tz="America/New_York") -> pd.DataFrame:
    """
    Pull 1-min OHLCV bars (IGPV-like) for a given ticker & trading date.
    trade_date: 'YYYY-MM-DD' in the exchange/local timezone.
    Returns a DataFrame indexed in local_tz with columns:
    ['open','high','low','close','volume','numEvents']
    """
    # Build local session day bounds (e.g., 09:30–16:00 ET for equities if desired).
    # Here we default to full calendar day in local_tz and let Bloomberg return what's available.
    dt_local = datetime.fromisoformat(trade_date).replace(tzinfo=ZoneInfo(local_tz))
    start_local = dt_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local   = start_local + timedelta(days=1)

    # Convert to UTC (API expects UTC timestamps)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc   = end_local.astimezone(timezone.utc)

    # Bloomberg session
    opts = SessionOptions()
    opts.setServerHost("localhost")
    opts.setServerPort(8194)

    session = Session(opts)
    if not session.start():
        raise RuntimeError("Failed to start Bloomberg session")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Failed to open //blp/refdata")

    svc = session.getService("//blp/refdata")
    req = svc.createRequest("IntradayBarRequest")
    req.set("security", ticker)
    req.set("eventType", "TRADE")        # trades like IGPV
    req.set("interval", int(interval))   # minutes
    req.set("startDateTime", start_utc)  # UTC
    req.set("endDateTime", end_utc)      # UTC
    # Optional: regularTradingHoursOnly, gapFillInitialBar, adjustmentNormal, adjustmentAbnormal
    # req.set("gapFillInitialBar", True)

    cid = blpapi.CorrelationId(1)
    session.sendRequest(req, correlationId=cid)

    rows = []
    while True:
        ev = session.nextEvent()
        for msg in ev:
            if msg.correlationIds() and msg.correlationIds()[0].value() == 1:
                if msg.messageType() == blpapi.Name("IntradayBarResponse"):
                    data = msg.getElement("barData").getElement("barTickData")
                    for i in range(data.numValues()):
                        bar = data.getValueAsElement(i)
                        ts_utc = bar.getElementAsDatetime("time").replace(tzinfo=timezone.utc)
                        ts_loc = ts_utc.astimezone(ZoneInfo(local_tz))
                        rows.append({
                            "time": ts_loc,
                            "open":  bar.getElementAsFloat("open"),
                            "high":  bar.getElementAsFloat("high"),
                            "low":   bar.getElementAsFloat("low"),
                            "close": bar.getElementAsFloat("close"),
                            "volume": bar.getElementAsInteger("volume"),
                            "numEvents": bar.getElementAsInteger("numEvents"),
                        })
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df

# Example:
# df = igpv_bars("AAPL US Equity", "2025-09-19", interval=1, local_tz="America/New_York")
# df.to_excel("AAPL_IGPV_2025-09-19.xlsx")
