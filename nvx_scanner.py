#!/usr/bin/env python3
"""Supply & Demand Zone Scanner, Nifty 500, built on the "Get Together Finance" notes.
Educational use only, not investment advice.

Rules implemented (from the notes):
  * Exciting candle = body > 50% of its range; base candle = body <= 50%.
  * Patterns: DBR, RBR (demand) / RBD, DBD (supply): legin -> base -> legout.
  * Marking (body to wick) + exceptional marking (legin wick for reversal, legout wick for all).
  * Closing concept: legout must close beyond the legin body and beyond the base.
  * Trade score: Freshness (3 / 1.5 / 0) + Strength (2 / 1) + Time at base (2 / 1 / 0) + EMA bonus (1).
    7+ = entry type 1, 5-6 = type 2/3, below 5 = no trade.
  * Trade setup: entry just beyond proximal, SL just beyond distal, target 2:1, check room to the target.
  * Trend: 50 SMA slope over 7 candles ("clock" rule). Location: curve analysis on the higher timeframe.
  * Timeframe table: execution TF -> intermediate (trend) TF -> higher (location) TF.

Setup:  pip install yfinance pandas numpy
Run:    python nvx_scanner.py              (daily/weekly/monthly/quarterly -> nvxscan.html)
        python nvx_scanner.py --intraday   (also 60/75/125 min, slower)
        python nvx_scanner.py --watch 30   (re-scan every 30 min)
"""
import argparse, io, json, time, urllib.request, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
DIST = {"60m": 6, "75m": 6, "125m": 6, "D": 6, "W": 10, "M": 15, "Q": 20, "H": 25, "Y": 30}   # max % distance kept per timeframe
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
GRADES = [(7, "A+"), (6, "A"), (5, "B"), (0, "C")]   # C = below 5 = no trade
TFS = ["60m", "75m", "125m", "D", "W", "M", "Q", "H", "Y"]     # low -> high
TT = {"60m": "60m", "75m": "75m", "125m": "125m", "D": "DIT", "W": "WIT", "M": "MIT", "Q": "QIT", "H": "HYIT", "Y": "YIT"}
NEAR_PCT = 1.0   # an HTF zone within 1% of price counts as confluence even if not overlapping
WMAX = {"60m": .03, "75m": .03, "125m": .04, "D": .06, "W": .10, "M": .15, "Q": .25, "H": .30, "Y": .40}
# execution TF -> (intermediate/trend TF, higher/location TF); W/M extend the notes' table by one step
MAP = {"60m": ("D", "W"), "75m": ("D", "W"), "125m": ("D", "W"), "D": ("W", "M"),
       "W": ("M", "Q"), "M": ("Q", "H"), "Q": ("H", "Y"), "H": ("Y", None), "Y": (None, None)}
OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
IMP_ATR = 1.0     # legout must be explosive: body >= 1.0 x ATR (adjustable)
LEGIN_ATR = 0.5   # legin body >= 0.5 x ATR
MAXBASE = 6       # base candles allowed (>5 scores 0 for time-at-base)
FLAT_ATR = 0.4    # 50 SMA change over 7 candles below 0.4 x ATR = sideways ("close to 3 o'clock")


def symbols():
    try:
        req = urllib.request.Request(LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=20).read()))
        df.to_csv("nifty500list.csv", index=False)
    except Exception as e:
        print("NSE list download failed, using local nifty500list.csv:", e)
        df = pd.read_csv("nifty500list.csv")
    return {s + ".NS": (s, n) for s, n in zip(df["Symbol"], df["Company Name"])}


def fetch(tickers, period, interval):
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        print(f"[{interval}] {i + 1}-{i + len(chunk)} of {len(tickers)}")
        try:
            d = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                            auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            print("  batch failed:", e)
            time.sleep(2)
            continue
        for t in chunk:
            try:
                x = d[t][["Open", "High", "Low", "Close"]].dropna()
                if len(x) > 30:
                    out[t] = x
            except Exception:
                pass
    return out


def rs(df, rules):
    for r in rules:
        try:
            return df.resample(r).agg(OHLC).dropna()
        except ValueError:
            continue
    return None


def sess(df, mins):
    """75/125-minute candles from 5-minute data, aligned to the NSE open (9:15)."""
    d = df.copy()
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    d.index = d.index.tz_convert("Asia/Kolkata")
    t = np.asarray(d.index.hour * 60 + d.index.minute - 555)
    m = (t >= 0) & (t < 375)
    d, t = d[m], t[m]
    keys = [np.asarray(d.index.date), t // mins]
    g = d.groupby(keys)
    out = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    ts = d.index.to_series().groupby(keys).first()
    out.index = pd.DatetimeIndex(list(ts))
    return out.dropna()


def half(df):
    keys = [df.index.year, (df.index.month > 6).astype(int)]
    out = df.groupby(keys).agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    ts = df.index.to_series().groupby(keys).first()
    out.index = pd.DatetimeIndex(list(ts))
    return out.dropna()


def build(daily, h60, m5):
    f = {}
    if daily is not None:
        f["D"] = daily
        for k, rules in (("W", ["W"]), ("M", ["ME", "M"]), ("Q", ["QE", "Q"]), ("Y", ["YE", "A", "Y"])):
            x = rs(daily, rules)
            if x is not None and len(x) > (16 if k in ("Q", "Y") else 25):
                f[k] = x
        hx = half(daily)
        if len(hx) > 16:
            f["H"] = hx
    if h60 is not None:
        f["60m"] = h60
    if m5 is not None:
        f["75m"], f["125m"] = sess(m5, 75), sess(m5, 125)
    return f


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def find_zones(df, wmax, complete=True):
    """legin (exciting) -> 1..6 base candles -> legout (exciting, explosive). Unbroken zones only."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    a, n = atr(df), len(df)
    body, rng = np.abs(c - o), h - l
    exc = body > 0.5 * rng                       # exciting candle: body > 50% of the candle
    zones = []
    for i in range(15, n if complete else n - 1):
        if np.isnan(a[i - 1]) or not exc[i] or body[i] < IMP_ATR * a[i - 1]:
            continue
        j = i - 1
        while j >= 0 and i - 1 - j < MAXBASE and not exc[j]:
            j -= 1
        nb = i - 1 - j
        if nb < 1 or j < 0 or not exc[j] or body[j] < LEGIN_ATR * a[i - 1]:
            continue
        b = slice(j + 1, i)
        lb, hb = l[b].min(), h[b].max()
        if c[i] > o[i]:                                       # demand: green legout
            if c[i] <= max(hb, o[j], c[j]):                   # closing concept
                continue
            kind, pat = "demand", ("DBR" if c[j] < o[j] else "RBR")
            prox = max(o[b].max(), c[b].max())                # highest body of the base
            dist = lb
            if pat == "DBR":
                dist = min(dist, l[j])                        # exceptional marking: legin wick
            dist = min(dist, l[i])                            # exceptional marking: legout wick
            gap = bool(l[i] > h[i - 1])
            two = bool(i + 1 < n and c[i + 1] > o[i + 1] and exc[i + 1])
            broken = bool((c[i + 1:] < dist).any())
            e = l[i + 1:] <= prox
        else:                                                 # supply: red legout
            if c[i] >= min(lb, o[j], c[j]):
                continue
            kind, pat = "supply", ("RBD" if c[j] > o[j] else "DBD")
            prox = min(o[b].min(), c[b].min())                # lowest body of the base
            dist = hb
            if pat == "RBD":
                dist = max(dist, h[j])
            dist = max(dist, h[i])
            gap = bool(h[i] < l[i - 1])
            two = bool(i + 1 < n and c[i + 1] < o[i + 1] and exc[i + 1])
            broken = bool((c[i + 1:] > dist).any())
            e = h[i + 1:] >= prox
        if broken or abs(prox - dist) / prox > wmax:
            continue
        visits = (int(e[0]) + int((e[1:] & ~e[:-1]).sum())) if len(e) else 0
        live = bool(len(e) and e[-1])                    # price is inside the zone right now
        touches = visits - 1 if live else visits         # the visit in progress is the first touch = still fresh
        pre_h, pre_l = h[max(0, j - 120):j], l[max(0, j - 120):j]
        beyond = bool(len(pre_h) and (prox > pre_h.max() if kind == "demand" else prox < pre_l.min()))
        zones.append(dict(i=i, j=j, kind=kind, pat=pat, prox=float(prox), dist=float(dist), nb=nb,
                          touches=touches, live=live, strength=2 if (gap or two) else 1, gap=gap,
                          garb=bool(nb >= 4 and beyond),                       # many base candles at new highs = poor zone
                          wide=bool(abs(prox - dist) > 1.5 * a[i - 1]),         # very wide base = imbalance, avoid
                          hi=beyond))                                          # zone at fresh highs/lows: wait for follow-up
    for z in zones:      # credibility: a zone overlapping an earlier same-type zone is a "reaction" zone
        lo, hi = sorted((z["prox"], z["dist"]))
        z["cred"] = not any(y["kind"] == z["kind"] and y["i"] < z["j"]
                            and min(y["prox"], y["dist"]) <= hi and max(y["prox"], y["dist"]) >= lo
                            for y in zones)
    return zones


def trend_of(df):
    """50 SMA over 7 candles: rising = up (12-3), falling = down (3-6), flat = sideways (close to 3). Returns (label, slope in ATR)."""
    if df is None or len(df) < 60:
        return None, None
    sma = df.Close.rolling(50).mean()
    d, a = float(sma.iloc[-1] - sma.iloc[-8]), atr(df)[-1]
    if np.isnan(d) or np.isnan(a) or a == 0:
        return None, None
    return ("side" if abs(d) < FLAT_ATR * a else ("up" if d > 0 else "down")), round(d / a, 2)


def location(df, zones, last):
    """Curve analysis: price position between nearest fresh demand and fresh supply (thirds). Returns (loc, demand zone, supply zone)."""
    if df is None:
        return "na", None, None
    dem = [z for z in zones if z["kind"] == "demand" and z["touches"] == 0 and z["prox"] <= last]
    sup = [z for z in zones if z["kind"] == "supply" and z["touches"] == 0 and z["prox"] >= last]
    dz = max(dem, key=lambda z: z["prox"]) if dem else None
    sz = min(sup, key=lambda z: z["prox"]) if sup else None
    dd = [round(dz["prox"], 2), round(dz["dist"], 2)] if dz else None
    ss = [round(sz["prox"], 2), round(sz["dist"], 2)] if sz else None
    if not dz or not sz or sz["prox"] <= dz["prox"]:
        return "eq", dd, ss                   # notes: missing fresh side = assume equilibrium
    p = (last - dz["prox"]) / (sz["prox"] - dz["prox"])
    return ("low" if p < 1 / 3 else "high" if p > 2 / 3 else "eq"), dd, ss


def rule(kind, loc, trend):
    want_loc, want_tr = ("low", "up") if kind == "demand" else ("high", "down")
    if loc == want_loc:
        return True
    if loc in ("eq", "na"):
        return None if trend is None else trend == want_tr
    return False


def grade(s):
    return next(g for m, g in GRADES if s >= m)


def last_complete(tf, df):
    """Daily candle is complete unless the market is open and the last bar is today's."""
    if tf != "D":
        return False
    now = dt.datetime.now(IST)
    forming = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) < (15, 30) and df.index[-1].date() == now.date()
    return not forming


def analyse(meta, f, aux=None):
    Z = {tf: find_zones(df, WMAX[tf], last_complete(tf, df)) for tf, df in f.items()}
    res = []
    for k, tf in enumerate(TFS):
        if tf not in f:
            continue
        df = f[tf]
        n, last = len(df), float(df.Close.values[-1])
        ema = [float(df.Close.ewm(span=s, adjust=False).mean().iloc[-1]) for s in (20, 50)]
        higher = [x for x in TFS[k + 1:] if x in f]
        itf, htf = MAP[tf]
        trend, tsl = trend_of(f.get(itf)) if itf else (None, None)
        loc, ld, ls = location(f.get(htf), Z.get(htf, []), last) if htf else ("na", None, None)
        fmt = "%d %b %y" if tf in ("D", "W", "M", "Q", "H", "Y") else "%d %b %H:%M"
        for kind in ("demand", "supply"):
            cand = []
            for z in Z[tf]:
                if z["kind"] != kind:
                    continue
                d = (last - z["prox"]) / last * 100 if kind == "demand" else (z["prox"] - last) / last * 100
                if -100 <= d <= DIST[tf]:
                    cand.append((abs(d), d, z))
            for _, d, z in sorted(cand, key=lambda x: x[0])[:2]:
                lo, hi = sorted((z["prox"], z["dist"]))
                tol = last * NEAR_PCT / 100
                conf = [x for x in higher if any(
                    w["kind"] == kind and w["touches"] == 0
                    and lo - tol <= max(w["prox"], w["dist"]) and hi + tol >= min(w["prox"], w["dist"])
                    for w in Z[x])]
                if kind == "demand":
                    entry, sl = z["prox"] * 1.0005, z["dist"] * 0.999
                    tgt = entry + 2 * (entry - sl)
                    opp = [y["prox"] for y in Z[tf] if y["kind"] == "supply" and y["prox"] > entry]
                    nxt = min(opp) if opp else None
                    room = nxt is None or nxt >= tgt
                else:
                    entry, sl = z["prox"] * 0.9995, z["dist"] * 1.001
                    tgt = entry - 2 * (sl - entry)
                    opp = [y["prox"] for y in Z[tf] if y["kind"] == "demand" and y["prox"] < entry]
                    nxt = max(opp) if opp else None
                    room = nxt is None or nxt <= tgt
                fresh = 3 if z["touches"] == 0 else 1.5 if z["touches"] == 1 else 0
                tbase = 2 if z["nb"] <= 3 else 1 if z["nb"] <= 5 else 0
                ema_ok = any(lo <= e <= hi for e in ema)
                score = fresh + z["strength"] + tbase + (1 if ema_ok else 0)
                want_loc, want_tr = ("low", "up") if kind == "demand" else ("high", "down")
                lok = True if loc == want_loc else (None if loc in ("eq", "na") else False)   # Location favourable / neutral / against
                tok = None if trend is None else (trend == want_tr)                           # Trend favourable (sideways = ignore)
                anchored = tf in ("Q", "H", "Y") or any(x in ("Q", "H", "Y") for x in conf)
                full = z["touches"] == 0 and not z["garb"] and anchored
                plan = "Full 2:1" if full else "Quick 1:1 (exit at entry if it reverses)"
                if kind == "demand":
                    t1, ep = entry + (entry - sl), [entry, (z["prox"] + z["dist"]) / 2, z["dist"] + 0.25 * (z["prox"] - z["dist"])]
                else:
                    t1, ep = entry - (sl - entry), [entry, (z["prox"] + z["dist"]) / 2, z["dist"] - 0.25 * (z["dist"] - z["prox"])]
                win = min(n, 100, max(60, n - 1 - z["j"] + 8))
                tail = df.tail(win)
                res.append(dict(
                    sym=meta[0], name=meta[1], tf=tf, tt=TT[tf], kind=kind, pat=z["pat"], garb=z["garb"], wide=z["wide"], hi=z["hi"], live=z["live"], grade=grade(score), score=score,
                    parts=[fresh, z["strength"], tbase, 1 if ema_ok else 0],
                    etype="Type 1" if score >= 7 else "Type 2/3" if score >= 5 else "No trade",
                    price=round(last, 2), prox=round(z["prox"], 2), dist=round(z["dist"], 2),
                    t1=round(float(t1), 2), ep=[round(float(v), 2) for v in ep], plan=plan,
                    entry=round(entry, 2), sl=round(sl, 2), target=round(float(tgt), 2),
                    nxt=None if nxt is None else round(float(nxt), 2), room=bool(room),
                    higher=higher, conf=conf, htf=htf, itf=itf, loc=loc, trend=trend,
                    rule=rule(kind, loc, trend), lok=lok, tok=tok, ld=ld, ls=ls, tsl=tsl, cred=bool(z["cred"]), nb=z["nb"], gap=z["gap"],
                    touches=z["touches"], age=n - 1 - z["i"], dist_pct=round(d, 2),
                    formed=df.index[z["i"]].strftime(fmt), idx=z["i"] - (n - win),
                    span=[df.index[-win].strftime(fmt), df.index[-1].strftime(fmt)],
                    lc=[df.index[-1].strftime(fmt)] + [round(float(v), 2) for v in df.iloc[-1][["Open", "High", "Low", "Close"]]],
                    candles=[[round(float(v), 2) for v in r] for r in tail[["Open", "High", "Low", "Close"]].values]))
    if aux is not None:      # candles + 50 SMA of the location / trend timeframes, stored once per stock
        for tf in {x for r in res for x in (r["htf"], r["itf"]) if x and x in f}:
            tl, sm = f[tf].tail(70), f[tf].Close.rolling(50).mean().tail(70)
            aux[meta[0] + "|" + tf] = dict(
                c=[[round(float(v), 2) for v in row] for row in tl[["Open", "High", "Low", "Close"]].values],
                s=[None if np.isnan(v) else round(float(v), 2) for v in sm.values],
                f=tl.index[0].strftime("%d %b %y"), l=tl.index[-1].strftime("%d %b %y"))
    return res


def drop_bad(frames):
    """Drop tickers whose data looks wrong: identical recent closes under two tickers (mixed data) or no recent candles."""
    seen, bad = {}, set()
    for t, df in frames.items():
        key = tuple(np.round(df.Close.values[-25:], 2))
        if len(key) < 25:
            continue
        if key in seen:
            bad.update({t, seen[key]})
        else:
            seen[key] = t
    if bad:
        print("Dropped (same prices under different tickers):", sorted(bad))
    try:
        newest = max(df.index[-1] for df in frames.values())
        stale = {t for t, df in frames.items() if (newest - df.index[-1]).days > 7}
        if stale:
            print("Dropped (no recent candles):", sorted(stale))
        bad |= stale
    except Exception:
        pass
    return {t: df for t, df in frames.items() if t not in bad}


def scan(intraday=False):
    meta = symbols()
    meta.update({"^NSEI": ("NIFTY50", "Nifty 50"), "^NSEBANK": ("BANKNIFTY", "Nifty Bank")})
    tk = list(meta)
    daily = drop_bad(fetch(tk, "max", "1d"))
    for smp in ("ONGC", "RELIANCE", "TCS", "HDFCBANK"):
        if smp + ".NS" in daily:
            print("SAMPLE", smp, daily[smp + ".NS"].index[-1].date(), "close", round(float(daily[smp + ".NS"].Close.iloc[-1]), 2))
    h60 = fetch(tk, "6mo", "60m") if intraday else {}
    m5 = fetch(tk, "60d", "5m") if intraday else {}
    out, ok, aux = [], 0, {}
    for t in tk:
        if t not in daily and t not in h60 and t not in m5:
            continue
        try:
            out += analyse(meta[t], build(daily.get(t), h60.get(t), m5.get(t)), aux)
            ok += 1
        except Exception as e:
            print("skip", t, e)
    asof = ""
    try:
        last = max(df.index[-1] for df in daily.values())
        now = dt.datetime.now(IST)
        forming = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) < (15, 30) and last.date() == now.date()
        asof = f"{last:%d %b %Y}" + (" (today's candle is still forming)" if forming else "")
    except Exception:
        pass
    print(f"{len(out)} setups from {ok} stocks, data till {asof}")
    return out, asof, aux


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zone Scanner</title><style>
:root{--bg:#fff;--fg:#14171a;--mut:#667085;--card:#f5f6f8;--bd:#e3e6ea;--g:#16a34a;--r:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#0f1317;--fg:#e8eaed;--mut:#98a2b3;--card:#1a2027;--bd:#2b333d}}
*{box-sizing:border-box}html,body{overflow-x:hidden}body{margin:0;padding:10px;font:14px/1.35 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:16px;margin:0}.mut{color:var(--mut);font-size:12px}.nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
button,select,input{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:6px 8px;min-height:38px;min-width:0}
.seg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:8px 0 6px}.seg button{font-weight:700}
.seg button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}.seg .d{color:var(--g)}.seg .s{color:var(--r)}.seg .on.d,.seg .on.s{color:var(--bg)}
.tfs{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:6px}.tfs button{display:flex;flex-direction:column;align-items:center;line-height:1.15;padding:4px 2px}
.tfs small{font-size:11px;color:var(--mut)}.tfs button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}.tfs button.on small{color:var(--bg)}
.row2{display:grid;grid-template-columns:1.1fr 1fr;gap:6px;margin-bottom:6px}.row3{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px}.row3 label{display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--mut)}.row3 select{width:100%}
#cnt{margin:4px 0 8px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:10px 11px;cursor:pointer;min-width:0}
.hd{display:flex;align-items:center;gap:6px}.sym{font-weight:700;font-size:15px}
.bd{font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid var(--bd);color:var(--mut)}
.side{font-weight:700;font-size:12px;margin-left:auto}.demand{color:var(--g)}.supply{color:var(--r)}
.gr{font-weight:700;font-size:12px;padding:1px 7px;border-radius:6px;background:var(--fg);color:var(--bg)}
.lv{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:8px;text-align:center}
.lv div{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:4px 2px;font-weight:600;font-size:13px}
.lv i{display:block;font-style:normal;font-weight:400;font-size:10px;color:var(--mut)}
.row{display:flex;justify-content:space-between;margin-top:6px}.tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.tag{font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--bd);color:var(--mut)}.tag.on{color:var(--g);border-color:var(--g)}.tag.off{color:var(--r);border-color:var(--r)}
.tabs{display:flex;gap:6px;margin:8px 0}.tab{flex:1}.tab.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}.tag.tap{cursor:pointer;padding:3px 8px;font-size:12px}
.more{display:block;margin:12px auto 20px}.empty{grid-column:1/-1;text-align:center;padding:30px 10px;color:var(--mut)}
details.help{margin:10px 0 24px}details.help summary{cursor:pointer;color:var(--mut);font-size:12px}details.help div{font-size:12px;color:var(--mut);margin-top:6px;line-height:1.6}
#m{position:fixed;inset:0;background:#0009;display:none;justify-content:center;padding:10px;overflow:auto;z-index:20}
#m>div{background:var(--bg);border-radius:14px;padding:12px;max-width:680px;width:100%;margin:auto}
.cap{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}.cap label{display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--mut)}
rect.demand{fill:#16a34a33}rect.supply{fill:#dc262633}svg{width:100%;height:auto}.x{float:right;min-height:32px;padding:2px 10px}
</style></head><body>
<h1>Zone Scanner · Nifty 500</h1><div class="mut">Last scan: __TIME__ IST <b id="age"></b> · Data till: __ASOF__</div><div id="warn" class="mut" style="color:var(--r)"></div><div class="mut">Educational only, not investment advice</div>
<div class="seg" id="side"></div>
<div class="tfs" id="tfs"></div>
<div class="row2"><input id="q" type="search" placeholder="Search stock">
<select id="sort"><option value="price">Price low→high</option><option value="pricedesc">Price high→low</option><option value="score">Best score</option><option value="near">Nearest zone</option></select></div>
<div class="row3"><label>Freshness<select id="fr"></select></label><label>Score / entry type<select id="sc"></select></label></div>
<div id="cnt" class="mut"></div><div id="grid"></div><button id="more" class="more" style="display:none"></button>
<details class="help"><summary>How to read this page</summary><div>
<b>Year / Half / Qtr / Month / Week / Day</b> = chart on which the zone is made (YIT, HYIT, QIT, MIT, WIT, DIT).<br>
<b>DBR, RBR</b> = demand (buy) zones, <b>RBD, DBD</b> = supply (sell) zones.<br>
<b>Freshness</b>: Fresh = price never came back to the zone, Tested 1× / 2×+ = price has returned that many times.<br><b>Score</b>: 7+ = Type 1 (set and forget), 5–6 = Type 2/3 (needs confirmation), below 5 = no trade. Grade A+ = 7+, A = 6, B = 5, C = below 5. The number after each option is how many setups it shows right now.<br>
<b>Entry</b> = just beyond the proximal line, <b>Stop loss</b> = just beyond the distal line. Tap a card for the chart, your Qty (set capital and risk there) and full details.<br>
<b>Location</b> (tap it) = where price sits between the nearest fresh demand and supply on the higher timeframe: Low favours buying, High favours selling. <b>Trend</b> (tap it) = 50 SMA slope on the middle timeframe: Up = buy, Down = sell, Sideways = ignore. <b>Execution</b> = the zone itself (trade score, tap it or the card). <b>★ Best</b> = all three are ✓ and score 7+.<br><b>Room ✓</b> = no opposite zone within 2x the stop-loss distance. <b>Price in zone</b> = price is at the zone now (first touch).<br>
<b>W ✓ / M ✓</b> = a fresh same-side zone on that higher timeframe overlaps this zone or is within 1% (confluence).</div></details>
<div id="m"><div id="mc"></div></div>
<script>
const D=__DATA__,AUX=__AUX__,LIM=__LIM__,$=id=>document.getElementById(id);
const N={Y:"Yearly",H:"Half-yearly",Q:"Quarterly",M:"Monthly",W:"Weekly",D:"Daily","125m":"125 min","75m":"75 min","60m":"60 min"},
SH={Y:"Year",H:"Half",Q:"Qtr",M:"Month",W:"Week",D:"Day","125m":"125m","75m":"75m","60m":"60m"},ORD=Object.keys(N),
LC={low:"Low",eq:"Equilibrium",high:"High",na:"n/a"},TR={up:"Up",down:"Down",side:"Sideways"},PAGE=40;
const CMP={price:(a,b)=>a.price-b.price||b.score-a.score,pricedesc:(a,b)=>b.price-a.price||b.score-a.score,
score:(a,b)=>b.score-a.score||b.conf.length-a.conf.length||a.price-b.price,near:(a,b)=>Math.abs(a.dist_pct)-Math.abs(b.dist_pct)||a.price-b.price};
const best=z=>z.score>=7&&z.lok===true&&z.tok===true;
const FR={"0":z=>z.touches==0,"1":z=>z.touches==1,"2":z=>z.touches>=2,all:()=>true},
SC={best:z=>best(z),"5":z=>z.score>=5,"7":z=>z.score>=7,t23:z=>z.score>=5&&z.score<7,no:z=>z.score<5,all:()=>true},
FRL=[["0","Fresh"],["1","Tested 1×"],["2","Tested 2×+"],["all","All zones"]],
SCL=[["best","★ Best (Loc+Trend+Exec)"],["5","Score 5+ (tradable)"],["7","7+ · Type 1"],["t23","5–6 · Type 2/3"],["no","Below 5 · no trade"],["all","All scores"]];
let S={side:"",tf:"ALL",sort:"price",fr:"0",sc:"5",cap:"100000",rk:"1"},L=[],shown=PAGE;
try{Object.assign(S,JSON.parse(localStorage.getItem("st3")||"{}"))}catch(e){}
if(S.tf!="ALL"&&!D.some(z=>z.tf==S.tf))S.tf="ALL";if(!FR[S.fr])S.fr="0";if(!SC[S.sc])S.sc="5";
const save=()=>{try{localStorage.setItem("st3",JSON.stringify(S))}catch(e){}};
const qty=z=>Math.floor((+S.cap||0)*(+S.rk)/100/Math.abs(z.entry-z.sl));
const chip=(v,a)=>v===true?`<span class="tag on">${a} ✓</span>`:v===false?`<span class="tag off">${a} ✗</span>`:"";
function match(z,q,skipTf,lim,st){st=st||S;return(skipTf||st.tf=="ALL"||z.tf==st.tf)&&z.dist_pct<=lim&&(!st.side||z.kind==st.side)&&FR[st.fr](z)&&SC[st.sc](z)&&(!q||(z.sym+" "+z.name).toLowerCase().includes(q))}
const tap=(z,i,mode,txt,v)=>`<span class="tag tap ${v===true?"on":v===false?"off":""}" onclick="event.stopPropagation();chart(${i},'${mode}')">${txt} ${v===true?"✓":v===false?"✗":"–"}</span>`;
function card(z,i){const tg=[best(z)?'<span class="tag on">★ Best setup</span>':'',tap(z,i,'loc','Location '+LC[z.loc],z.lok),tap(z,i,'trend','Trend '+(z.trend?TR[z.trend]:'–'),z.tok),tap(z,i,'exec','Execution '+z.score,z.score>=7?true:(z.score<5?false:null)),z.live?'<span class="tag on">Price in zone</span>':'',z.touches?`<span class="tag off">Tested ${z.touches}</span>`:'<span class="tag on">Fresh</span>',chip(z.room,"Room")];
z.conf.forEach(x=>tg.push(`<span class="tag on">${SH[x]} ✓</span>`));
if(!z.cred)tg.push('<span class="tag off">Reaction zone</span>');if(z.garb)tg.push('<span class="tag off">Poor base</span>');if(z.wide)tg.push('<span class="tag off">Wide base</span>');
return `<div class="card" onclick="chart(${i})"><div class="hd"><span class="sym">${z.sym}</span><span class="bd">${z.tt}</span><span class="side ${z.kind}">${z.kind=="demand"?"DEMAND":"SUPPLY"} ${z.pat}</span><span class="gr">${z.grade}</span></div>
<div class="mut nm">₹${z.price} · ${z.dist_pct}% away · ${z.name}</div>
<div class="lv"><div>${z.entry}<i>Entry</i></div><div class="supply">${z.sl}<i>Stop loss</i></div><div>${qty(z)}<i>Qty</i></div></div>
<div class="row"><span class="mut">Score ${z.score} · ${z.etype}</span><span class="mut">${z.plan.split(" (")[0]}</span></div><div class="tags">${tg.join("")}</div></div>`}
function draw(){const q=$("q").value.trim().toLowerCase(),cnt={};let tot=0;D.forEach(z=>{if(match(z,q,true,LIM[z.tf]))cnt[z.tf]=(cnt[z.tf]||0)+1;if(match(z,q,true,LIM.D))tot++});
$("side").innerHTML=[["demand","Demand (Buy)","d"],["supply","Supply (Sell)","s"],["","Both",""]].map(([k,n,c])=>`<button class="${c} ${S.side==k?"on":""}" onclick="setSide('${k}')">${n}</button>`).join("");
$("tfs").innerHTML=[["ALL","All",tot]].concat(ORD.filter(k=>D.some(z=>z.tf==k)).map(k=>[k,SH[k],cnt[k]||0])).map(([k,n,c])=>`<button class="${S.tf==k?"on":""}" onclick="setTf('${k}')"><b>${n}</b><small>${c}</small></button>`).join("");
const lim=S.tf=="ALL"?LIM.D:LIM[S.tf];L=D.filter(z=>match(z,q,false,lim)).sort(CMP[S.sort]||CMP.price);
const opt=(arr,key)=>arr.map(([v,n])=>`<option value="${v}">${n} · ${D.filter(z=>match(z,q,false,lim,{...S,[key]:v})).length}</option>`).join("");
$("fr").innerHTML=opt(FRL,"fr");$("sc").innerHTML=opt(SCL,"sc");$("fr").value=S.fr;$("sc").value=S.sc;
$("cnt").textContent=L.length?`Showing ${Math.min(shown,L.length)} of ${L.length} setups · zones within ${lim}% of price`:"";
$("grid").innerHTML=L.length?L.slice(0,shown).map(card).join(""):'<div class="empty">No setups match.<br><br><button onclick="resetAll()">Reset</button></div>';
$("more").style.display=L.length>shown?"block":"none";$("more").textContent=`Show more (${L.length-shown} left)`}
function sync(){$("sort").value=S.sort;$("fr").value=S.fr;$("sc").value=S.sc}
function setTf(k){S.tf=k;shown=PAGE;save();draw()}
function setSide(k){S.side=k;shown=PAGE;save();draw()}
function resetAll(){S.side="";S.tf="ALL";S.sort="price";S.fr="0";S.sc="5";$("q").value="";shown=PAGE;sync();save();draw()}
function closeM(){$("m").style.display="none"}
function capChange(i){S.cap=$("cp").value;S.rk=$("rk").value;save();$("qt").textContent=qty(L[i]);draw()}
function cs(c,w,y){return c.map((k,j)=>{const cc=k[3]>=k[0]?"#16a34a":"#dc2626",x=j*w+w/2;return `<line x1="${x}" x2="${x}" y1="${y(k[1])}" y2="${y(k[2])}" stroke="${cc}"/><rect x="${x-w*.35}" y="${y(Math.max(k[0],k[3]))}" width="${w*.7}" height="${Math.max(1,Math.abs(y(k[0])-y(k[3])))}" fill="${cc}"/>`}).join("")}
function execView(z,i){const c=z.candles,W=640,H=290,n=c.length,w=W/n,bs=Math.max(0,z.idx-z.nb);
const lo=Math.min(...c.map(x=>x[2]),z.sl,z.dist)*.997,hi=Math.max(...c.map(x=>x[1]))*1.003,y=v=>H-6-(v-lo)/(hi-lo)*(H-26),
col=z.kind=="demand"?"#16a34a":"#dc2626",ty=Math.min(y(z.prox),y(z.dist)),bh=Math.abs(y(z.prox)-y(z.dist)),pu=z.prox>z.dist;
let s=`<svg viewBox="0 0 ${W} ${H}"><rect class="${z.kind}" x="${bs*w}" y="${ty}" width="${W-bs*w}" height="${Math.max(2,bh)}"/>`+cs(c,w,y);
s+=`<line x1="0" x2="${W}" y1="${y(z.entry)}" y2="${y(z.entry)}" stroke="#2563eb" stroke-dasharray="5"/><text x="4" y="${y(z.entry)-3}" font-size="11" fill="#2563eb">Entry ${z.entry}</text>
<line x1="0" x2="${W}" y1="${y(z.sl)}" y2="${y(z.sl)}" stroke="#dc2626" stroke-dasharray="5"/><text x="4" y="${y(z.sl)+11}" font-size="11" fill="#dc2626">SL ${z.sl}</text>
<text x="${W-4}" y="${pu?y(z.prox)-3:y(z.prox)+10}" font-size="9" text-anchor="end" fill="${col}">Proximal ${z.prox}</text><text x="${W-4}" y="${pu?y(z.dist)+10:y(z.dist)-3}" font-size="9" text-anchor="end" fill="${col}">Distal ${z.dist}</text></svg>`;
const p=z.parts;
return `<div class="mut">${N[z.tf]} candles ${z.span[0]} → ${z.span[1]} · last candle ${z.lc[0]}: O ${z.lc[1]} H ${z.lc[2]} L ${z.lc[3]} C ${z.lc[4]} (match with the ${N[z.tf]} chart in your app)</div>${s}
<div class="cap"><label>Capital (₹)<input id="cp" type="number" inputmode="numeric" value="${S.cap}" oninput="capChange(${i})"></label>
<label>Risk per trade<select id="rk" onchange="capChange(${i})"><option value="1">1% (beginner)</option><option value="1.5">1.5%</option><option value="2">2% (pro)</option></select></label></div>
<div><b>Qty <span id="qt">${qty(z)}</span></b> <span class="mut">at entry ${z.entry}, SL ${z.sl}</span></div>
<div class="mut">Score ${z.score} = Fresh ${p[0]} + Strength ${p[1]} + Base ${p[2]} + EMA ${p[3]} → ${z.etype} · base candles ${z.nb}${z.gap?" · gap":""}</div>
<div class="mut">Plan: ${z.plan}</div><div class="mut">Entry ladder EP1 ${z.ep[0]} / EP2 ${z.ep[1]} / EP3 ${z.ep[2]} · Proximal ${z.prox} · Distal ${z.dist}</div>
<div class="mut">Confirmed by: ${z.conf.map(x=>N[x]).join(", ")||"none"} · Opposite zone: ${z.nxt??"none"} · ${z.cred?"Credible":"Reaction of an earlier zone (non-reliable)"}</div>`}
const NA=t=>`<div class="mut" style="margin-top:8px">${t}</div>`;
function locView(z){if(!z.htf)return NA("This zone is on the highest timeframe, so Location is not checked.");
const a=AUX[z.sym+"|"+z.htf];if(!a)return NA("Chart data for the "+N[z.htf]+" timeframe is not available.");
const c=a.c,W=640,H=300,n=c.length,w=W/n,d=z.ld,s=z.ls;
const vals=c.flatMap(k=>[k[1],k[2]]).concat([z.price],d||[],s||[]),lo=Math.min(...vals)*.995,hi=Math.max(...vals)*1.005,y=v=>H-6-(v-lo)/(hi-lo)*(H-14);
let g=`<svg viewBox="0 0 ${W} ${H}">`,both=d&&s&&s[0]>d[0];
if(both){const t1=d[0]+(s[0]-d[0])/3,t2=d[0]+(s[0]-d[0])*2/3;
g+=`<rect x="0" y="${y(t1)}" width="${W}" height="${y(d[0])-y(t1)}" fill="#16a34a" opacity=".10"/><rect x="0" y="${y(t2)}" width="${W}" height="${y(t1)-y(t2)}" fill="#888" opacity=".10"/><rect x="0" y="${y(s[0])}" width="${W}" height="${y(t2)-y(s[0])}" fill="#dc2626" opacity=".10"/>
<text x="${W-4}" y="${y(d[0])-4}" font-size="10" text-anchor="end" fill="#16a34a">Low</text><text x="${W-4}" y="${y(t1)-4}" font-size="10" text-anchor="end" fill="#888">Equilibrium</text><text x="${W-4}" y="${y(t2)-4}" font-size="10" text-anchor="end" fill="#dc2626">High</text>`}
g+=cs(c,w,y);
const zb=(zz,col,t)=>zz?`<rect x="0" y="${Math.min(y(zz[0]),y(zz[1]))}" width="${W}" height="${Math.max(2,Math.abs(y(zz[0])-y(zz[1])))}" fill="${col}" opacity=".28"/><text x="4" y="${Math.min(y(zz[0]),y(zz[1]))-3}" font-size="10" fill="${col}">${t} ${zz[0]}–${zz[1]}</text>`:"";
g+=zb(d,"#16a34a","Fresh demand")+zb(s,"#dc2626","Fresh supply")+`<line x1="0" x2="${W}" y1="${y(z.price)}" y2="${y(z.price)}" stroke="#2563eb" stroke-dasharray="5"/><text x="4" y="${y(z.price)+12}" font-size="11" fill="#2563eb">Price ${z.price}</text></svg>`;
const pos=both?Math.round((z.price-d[0])/(s[0]-d[0])*100):null,fav=z.lok===true?"favourable":z.lok===false?"against this setup":"neutral (trend decides)";
return `<div class="mut">Location is read on the ${N[z.htf]} chart (${a.f} → ${a.l})</div>${g}
<div class="mut">Nearest fresh demand: ${d?d[0]+" – "+d[1]:"none"} · Nearest fresh supply: ${s?s[0]+" – "+s[1]:"none"}</div>
<div class="mut">${pos===null?"One side has no fresh zone, so the notes say to treat price as Equilibrium and let the trend decide.":"Price is "+pos+"% of the way from demand to supply (split in thirds: Low / Equilibrium / High)."}</div>
<div><b>Location: ${LC[z.loc]}</b> → ${fav} for a ${z.kind} setup. <span class="mut">Low favours buying at demand, High favours selling at supply.</span></div>`}
function trendView(z){if(!z.itf)return NA("This zone is on the highest timeframe, so Trend is not checked.");
const a=AUX[z.sym+"|"+z.itf];if(!a)return NA("Chart data for the "+N[z.itf]+" timeframe is not available.");
const c=a.c,sm=a.s,W=640,H=300,n=c.length,w=W/n,vals=c.flatMap(k=>[k[1],k[2]]).concat(sm.filter(v=>v!==null)),lo=Math.min(...vals)*.995,hi=Math.max(...vals)*1.005,y=v=>H-6-(v-lo)/(hi-lo)*(H-14);
let g=`<svg viewBox="0 0 ${W} ${H}">`+cs(c,w,y);
g+=`<polyline points="${sm.map((v,j)=>v===null?"":(j*w+w/2)+","+y(v)).filter(Boolean).join(" ")}" fill="none" stroke="#f59e0b" stroke-width="2"/>`;
const j7=n-8,v7=sm[j7],v0=sm[n-1];
if(v7!==null&&v7!==undefined&&v0!==null){const x7=j7*w+w/2;g+=`<line x1="${x7}" x2="${x7}" y1="0" y2="${H}" stroke="#888" stroke-dasharray="4"/><line x1="${x7}" x2="${W}" y1="${y(v7)}" y2="${y(v7)}" stroke="#888" stroke-dasharray="4"/><text x="${x7+3}" y="12" font-size="10" fill="#888">7 candles back</text><circle cx="${n*w-w/2}" cy="${y(v0)}" r="3.5" fill="#f59e0b"/>`}
g+="</svg>";const fav=z.tok===true?"favours this setup":z.tok===false?"does not favour this setup":"unknown";
return `<div class="mut">Trend is read on the ${N[z.itf]} chart (${a.f} → ${a.l}) with the 50 SMA (orange)</div>${g}
<div class="mut">The 50 SMA moved ${z.tsl==null?"–":(z.tsl>0?"+":"")+z.tsl} × ATR over the last 7 candles (flat if between ±0.4).</div>
<div><b>Trend: ${z.trend?TR[z.trend]:"–"}</b> → ${fav} for a ${z.kind} setup. <span class="mut">Up = buy at demand, Down = sell at supply, Sideways = ignore.</span></div>`}
function chart(i,mode){mode=mode||"exec";const z=L[i];
const tabs=[["exec","Execution"],["loc","Location"],["trend","Trend"]].map(([k,n])=>`<button class="tab ${mode==k?"on":""}" onclick="chart(${i},'${k}')">${n}</button>`).join("");
const body=mode=="loc"?locView(z):mode=="trend"?trendView(z):execView(z,i);
$("mc").innerHTML=`<button class="x" onclick="closeM()">✕</button><div class="sym">${z.sym} · ${z.tt} (${N[z.tf]||z.tf})</div><div class="mut">${z.kind} ${z.pat} · grade ${z.grade}${best(z)?" · ★ Best setup":""} · formed ${z.formed} · price ${z.price}</div><div class="tabs">${tabs}</div>${body}`;
if(mode=="exec")$("rk").value=S.rk;$("m").style.display="flex"}
$("sort").onchange=()=>{S.sort=$("sort").value;shown=PAGE;save();draw()};$("fr").onchange=()=>{S.fr=$("fr").value;shown=PAGE;save();draw()};$("sc").onchange=()=>{S.sc=$("sc").value;shown=PAGE;save();draw()};
$("q").oninput=()=>{shown=PAGE;draw()};$("more").onclick=()=>{shown+=PAGE;draw()};$("m").onclick=e=>{if(e.target==$("m"))closeM()};
(function(){const m=Math.max(0,Math.round((Date.now()-new Date("__TS__"))/60000)),t=m<90?m+" min":(m/60).toFixed(1)+" h",
ist=new Date(Date.now()+19800000),mins=ist.getUTCHours()*60+ist.getUTCMinutes(),open=ist.getUTCDay()>=1&&ist.getUTCDay()<=5&&mins>=555&&mins<=930;
$("age").textContent="("+t+" ago)";
if(open&&m>75)$("warn").textContent="⚠ Market is open but this scan is "+t+" old. Run the scan again: GitHub → Actions → scan → Run workflow.";})();
sync();draw();
</script></body></html>"""


def write_html(data, path, asof="", aux=None):
    ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__AUX__", json.dumps(aux or {}, separators=(",", ":"))).replace("__TIME__", ist.strftime("%d %b %Y %H:%M")).replace("__ASOF__", asof or "n/a").replace("__TS__", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")).replace("__LIM__", json.dumps(DIST))
    open(path, "w", encoding="utf-8").write(html)
    print("Saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="re-scan every N minutes")
    ap.add_argument("--out", default="nvxscan.html")
    ap.add_argument("--intraday", action="store_true", help="also scan 60/75/125 min (slower)")
    a = ap.parse_args()
    while True:
        data, asof, aux = scan(a.intraday)
        write_html(data, a.out, asof, aux)
        if not a.watch:
            break
        time.sleep(a.watch * 60)
