#!/usr/bin/env python3
"""Multi-timeframe Supply & Demand Zone Scanner, Nifty 500 (educational, not advice).

Timeframes: 60m, 75m, 125m, Daily, Weekly, Monthly, Quarterly.
Confluence: a zone is "confirmed" by a higher timeframe when a same-type zone there overlaps it.

Setup:  pip install yfinance pandas numpy
Run:    python nvx_scanner.py              (one scan -> nvxscan.html)
        python nvx_scanner.py --watch 30   (re-scan every 30 min)
"""
import argparse, io, json, time, urllib.request, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
MAX_DIST = 6.0
GRADES = [(11, "A+"), (8, "A"), (5, "B"), (0, "C")]
TFS = ["60m", "75m", "125m", "D", "W", "M", "Q"]          # low -> high
WMAX = {"60m": .03, "75m": .03, "125m": .04, "D": .06, "W": .10, "M": .15, "Q": .25}
AGE_OK = {"60m": 60, "75m": 60, "125m": 60, "D": 60, "W": 26, "M": 12, "Q": 6}
OHLC = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}


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
                            auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print("  batch failed:", e)
            time.sleep(2)
            continue
        for t in chunk:
            try:
                x = d[t].dropna()
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
    """Build 75/125-minute candles from 5-minute data, aligned to NSE open (9:15)."""
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


def build(daily, h60, m5):
    f = {}
    if daily is not None:
        f["D"] = daily
        for k, rules in (("W", ["W"]), ("M", ["ME", "M"]), ("Q", ["QE", "Q"])):
            x = rs(daily, rules)
            if x is not None and len(x) > 25:
                f[k] = x
    if h60 is not None:
        f["60m"] = h60
    if m5 is not None:
        f["75m"], f["125m"] = sess(m5, 75), sess(m5, 125)
    return f


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def find_zones(df, wmax):
    """Base of 1-3 small candles followed by a strong impulse candle; unbroken zones only."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    a, n = atr(df), len(df)
    body, rng = np.abs(c - o), h - l
    zones = []
    for i in range(20, n):
        if np.isnan(a[i - 1]) or body[i] < 1.5 * a[i - 1] or body[i] < 0.6 * rng[i]:
            continue
        j = i - 1
        while j >= i - 3 and j >= 0 and body[j] <= 0.5 * a[i - 1]:
            j -= 1
        if j + 1 == i:
            continue
        b = slice(j + 1, i)
        up = c[i] > o[i]
        if (up and c[i] <= h[b].max()) or ((not up) and c[i] >= l[b].min()):
            continue
        bt, bb = np.maximum(o[b], c[b]), np.minimum(o[b], c[b])
        if up:
            prox, dist, kind = bt.max(), l[b].min(), "demand"
            broken = (c[i + 1:] < dist).any()
            e = l[i + 1:] <= prox
        else:
            prox, dist, kind = bb.min(), h[b].max(), "supply"
            broken = (c[i + 1:] > dist).any()
            e = h[i + 1:] >= prox
        if broken or abs(prox - dist) / prox > wmax:
            continue
        touches = (int(e[0]) + int((e[1:] & ~e[:-1]).sum())) if len(e) else 0
        zones.append(dict(i=i, kind=kind, prox=float(prox), dist=float(dist),
                          imp=float(body[i] / a[i - 1]), nbase=i - (j + 1), touches=touches))
    return zones


def grade(s):
    return next(g for m, g in GRADES if s >= m)


def analyse(meta, f):
    trend_up = trend_dn = False
    dly = f.get("D")
    if dly is not None and len(dly) > 200:
        c = dly.Close
        s50, s200, lc = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1], c.iloc[-1]
        trend_up, trend_dn = bool(lc > s50 > s200), bool(lc < s50 < s200)
    Z = {tf: find_zones(df, WMAX[tf]) for tf, df in f.items()}
    res = []
    for k, tf in enumerate(TFS):
        if tf not in f:
            continue
        df = f[tf]
        n, last = len(df), float(df.Close.values[-1])
        h, lo_ = df.High.values, df.Low.values
        higher = [x for x in TFS[k + 1:] if x in f]
        fmt = "%d %b %y" if tf in ("D", "W", "M", "Q") else "%d %b %H:%M"
        for kind in ("demand", "supply"):
            cand = []
            for z in Z[tf]:
                if z["kind"] != kind:
                    continue
                d = (last - z["prox"]) / last * 100 if kind == "demand" else (z["prox"] - last) / last * 100
                if -100 <= d <= MAX_DIST:
                    cand.append((abs(d), d, z))
            for _, d, z in sorted(cand, key=lambda x: x[0])[:2]:
                lo, hi = sorted((z["prox"], z["dist"]))
                conf = [x for x in higher if any(
                    w["kind"] == kind and lo <= max(w["prox"], w["dist"]) and hi >= min(w["prox"], w["dist"])
                    for w in Z[x])]
                trend = trend_up if kind == "demand" else trend_dn
                entry = z["prox"]
                if kind == "demand":
                    sl = z["dist"] * 0.999
                    risk = entry - sl
                    top = float(h[-120:].max())
                    tgt = top if top > entry + 1.5 * risk else entry + 2 * risk
                else:
                    sl = z["dist"] * 1.001
                    risk = sl - entry
                    bot = float(lo_[-120:].min())
                    tgt = bot if bot < entry - 1.5 * risk else entry - 2 * risk
                rr = abs(tgt - entry) / risk
                age = n - 1 - z["i"]
                score = (3 if z["touches"] == 0 else 1 if z["touches"] == 1 else 0) \
                    + (2 if z["imp"] >= 2.5 else 1) + (1 if z["nbase"] <= 2 else 0) \
                    + 2 * min(len(conf), 3) + (1 if trend else 0) + (1 if rr >= 3 else 0) \
                    + (1 if age <= AGE_OK[tf] else 0)
                tail = df.tail(60)
                res.append(dict(
                    sym=meta[0], name=meta[1], tf=tf, kind=kind, grade=grade(score), score=score,
                    price=round(last, 2), prox=round(z["prox"], 2), dist=round(z["dist"], 2),
                    entry=round(entry, 2), sl=round(sl, 2), target=round(float(tgt), 2), rr=round(rr, 2),
                    higher=higher, conf=conf, trend=bool(trend), age=age, touches=z["touches"],
                    dist_pct=round(d, 2), formed=df.index[z["i"]].strftime(fmt), idx=z["i"] - (n - 60),
                    candles=[[round(float(v), 2) for v in r] for r in tail[["Open", "High", "Low", "Close"]].values]))
    return res


def scan(intraday=False):
    meta = symbols()
    tk = list(meta)
    daily = fetch(tk, "10y", "1d")
    h60 = fetch(tk, "6mo", "60m") if intraday else {}
    m5 = fetch(tk, "60d", "5m") if intraday else {}
    out, ok = [], 0
    for t in tk:
        if t not in daily and t not in h60 and t not in m5:
            continue
        try:
            f = build(daily.get(t), h60.get(t), m5.get(t))
            out += analyse(meta[t], f)
            ok += 1
        except Exception as e:
            print("skip", t, e)
    print(f"{len(out)} setups from {ok} stocks")
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NVX Zone Scanner</title><style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#6b7280;--card:#f4f5f7;--bd:#e2e4e8;--g:#16a34a;--r:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;--card:#1b2027;--bd:#2c333c}}
body{margin:0;padding:12px;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 2px}.mut{color:var(--mut);font-size:12px}
button,select,input{font:inherit;padding:7px 10px;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:8px}
button.on{background:var(--fg);color:var(--bg)}.bar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px;cursor:pointer}
.row{display:flex;justify-content:space-between;margin-top:4px}.b{font-weight:600}
.demand{color:var(--g)}.supply{color:var(--r)}.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.tag{font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--bd);color:var(--mut)}.tag.on{color:var(--g);border-color:var(--g)}
#m{position:fixed;inset:0;background:#0008;display:none;align-items:center;justify-content:center;padding:10px}
#m>div{background:var(--bg);border-radius:12px;padding:12px;max-width:680px;width:100%}
rect.demand{fill:#16a34a33}rect.supply{fill:#dc262633}svg{width:100%;height:auto}
</style></head><body>
<h1>NVX Zone Scanner · Nifty 500</h1><div class="mut">Last scan: __TIME__ IST · educational only, not investment advice</div>
<div class="mut">Confluence = same-type zone on a higher timeframe overlapping this zone (✓ = confirmed, ✗ = not)</div>
<div class="bar" id="tfs"></div>
<div class="bar"><select id="side"><option value="">Demand + Supply</option><option value="demand">Demand</option><option value="supply">Supply</option></select>
<select id="grade"><option value="">Any grade</option><option>A+</option><option>A</option><option>B</option></select>
<select id="cf"><option value="0">Any confluence</option><option value="1">1+ higher TF</option><option value="2">2+ higher TF</option><option value="3">3+ higher TF</option></select>
<select id="tr"><option value="">Any trend</option><option value="1">Trend ✓ only</option></select>
<select id="px"><option value="6">Within 6%</option><option value="3">Within 3%</option><option value="1">Within 1%</option></select>
<input id="q" placeholder="Search stock"><span id="cnt" class="mut" style="align-self:center"></span></div>
<div id="grid"></div><div id="m" onclick="this.style.display='none'"><div id="mc"></div></div>
<script>
const D=__DATA__,$=id=>document.getElementById(id),R={"A+":4,A:3,B:2,C:1},
N={"60m":"60 min","75m":"75 min","125m":"125 min",D:"Daily",W:"Weekly",M:"Monthly",Q:"Quarterly"};let tf="D",L=[];
Object.keys(N).filter(k=>D.some(z=>z.tf==k)).forEach(k=>{const b=document.createElement("button");b.textContent=N[k];if(k==tf)b.classList.add("on");
b.onclick=()=>{tf=k;[...$("tfs").children].forEach(x=>x.classList.toggle("on",x==b));draw()};$("tfs").appendChild(b)});
const chips=z=>z.higher.length?z.higher.map(x=>`<span class="tag ${z.conf.includes(x)?"on":""}">${x} ${z.conf.includes(x)?"✓":"✗"}</span>`).join(""):'<span class="tag">Highest TF</span>';
function draw(){const s=$("side").value,g=R[$("grade").value]||0,q=$("q").value.toLowerCase(),px=+$("px").value,cf=+$("cf").value,tr=$("tr").value;
L=D.filter(z=>z.tf==tf&&(!s||z.kind==s)&&R[z.grade]>=g&&z.conf.length>=cf&&(!tr||z.trend)&&Math.abs(z.dist_pct)<=px&&(z.sym+z.name).toLowerCase().includes(q))
.sort((a,b)=>b.score-a.score||Math.abs(a.dist_pct)-Math.abs(b.dist_pct));
$("cnt").textContent=L.length+" setups";
$("grid").innerHTML=L.map((z,i)=>`<div class="card" onclick="chart(${i})"><div class="row"><span class="b">${z.sym}</span><span class="b ${z.kind}">${z.kind.toUpperCase()} · ${z.grade}</span></div>
<div class="mut">${z.name}</div><div class="row"><span>Entry ${z.entry}</span><span>SL ${z.sl}</span></div><div class="row"><span>Target ${z.target}</span><span>RR ${z.rr}</span></div>
<div class="mut" style="margin-top:6px">Confluence ${z.conf.length}/${z.higher.length}</div><div class="chips">${chips(z)}</div>
<div class="chips"><span class="tag">${z.trend?"Trend ✓":"Trend ✗"}</span><span class="tag">Age ${z.age} bars</span><span class="tag">${z.dist_pct}% away</span></div></div>`).join("")}
function chart(i){const z=L[i],c=z.candles,W=640,H=280,n=c.length,w=W/n;
const lo=Math.min(...c.map(x=>x[2]),z.sl)*.995,hi=Math.max(...c.map(x=>x[1]),z.target)*1.005,y=v=>H-(v-lo)/(hi-lo)*H;
const x0=Math.max(0,z.idx)*w;let s=`<svg viewBox="0 0 ${W} ${H}"><rect class="${z.kind}" x="${x0}" y="${y(Math.max(z.prox,z.dist))}" width="${W-x0}" height="${Math.abs(y(z.prox)-y(z.dist))}"/>`;
c.forEach((k,j)=>{const col=k[3]>=k[0]?"#16a34a":"#dc2626",x=j*w+w/2;
s+=`<line x1="${x}" x2="${x}" y1="${y(k[1])}" y2="${y(k[2])}" stroke="${col}"/><rect x="${x-w*.35}" y="${y(Math.max(k[0],k[3]))}" width="${w*.7}" height="${Math.max(1,Math.abs(y(k[0])-y(k[3])))}" fill="${col}"/>`});
[[z.entry,"#2563eb","Entry"],[z.sl,"#dc2626","SL"],[z.target,"#16a34a","Target"]].forEach(([v,col,t])=>s+=`<line x1="0" x2="${W}" y1="${y(v)}" y2="${y(v)}" stroke="${col}" stroke-dasharray="4"/><text x="4" y="${y(v)-3}" font-size="11" fill="${col}">${t} ${v}</text>`);
$("mc").innerHTML=`<div class="b">${z.sym} · ${N[z.tf]} · ${z.kind} · grade ${z.grade} (score ${z.score})</div><div class="mut">Formed ${z.formed} · touches ${z.touches} · price ${z.price}</div>
<div class="mut">Confirmed by: ${z.conf.map(x=>N[x]).join(", ")||"none"}</div>${s}</svg>`;$("m").style.display="flex"}
["side","grade","cf","tr","px","q"].forEach(id=>$(id).oninput=draw);draw();
</script></body></html>"""


def write_html(data, path):
    ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    html = PAGE.replace("__DATA__", json.dumps(data)).replace("__TIME__", ist.strftime("%d %b %Y %H:%M"))
    open(path, "w", encoding="utf-8").write(html)
    print("Saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="re-scan every N minutes")
    ap.add_argument("--out", default="nvxscan.html")
    ap.add_argument("--intraday", action="store_true", help="also scan 60/75/125 min (slower)")
    a = ap.parse_args()
    while True:
        write_html(scan(a.intraday), a.out)
        if not a.watch:
            break
        time.sleep(a.watch * 60)
