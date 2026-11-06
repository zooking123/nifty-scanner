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
MAX_DIST = 6.0                                  # show zones within 6% of price
GRADES = [(7, "A+"), (6, "A"), (5, "B"), (0, "C")]   # C = below 5 = no trade
TFS = ["60m", "75m", "125m", "D", "W", "M", "Q"]     # low -> high
WMAX = {"60m": .03, "75m": .03, "125m": .04, "D": .06, "W": .10, "M": .15, "Q": .25}
# execution TF -> (intermediate/trend TF, higher/location TF); W/M extend the notes' table by one step
MAP = {"60m": ("D", "W"), "75m": ("D", "W"), "125m": ("D", "W"), "D": ("W", "M"),
       "W": ("M", "Q"), "M": ("Q", None), "Q": (None, None)}
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
    """legin (exciting) -> 1..6 base candles -> legout (exciting, explosive). Unbroken zones only."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    a, n = atr(df), len(df)
    body, rng = np.abs(c - o), h - l
    exc = body > 0.5 * rng                       # exciting candle: body > 50% of the candle
    zones = []
    for i in range(21, n):
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
        touches = (int(e[0]) + int((e[1:] & ~e[:-1]).sum())) if len(e) else 0
        zones.append(dict(i=i, j=j, kind=kind, pat=pat, prox=float(prox), dist=float(dist), nb=nb,
                          touches=touches, strength=2 if (gap or two) else 1, gap=gap))
    for z in zones:      # credibility: a zone overlapping an earlier same-type zone is a "reaction" zone
        lo, hi = sorted((z["prox"], z["dist"]))
        z["cred"] = not any(y["kind"] == z["kind"] and y["i"] < z["j"]
                            and min(y["prox"], y["dist"]) <= hi and max(y["prox"], y["dist"]) >= lo
                            for y in zones)
    return zones


def trend_of(df):
    """50 SMA over 7 candles: rising = up (12-3), falling = down (3-6), flat = sideways (close to 3)."""
    if df is None or len(df) < 60:
        return None
    sma = df.Close.rolling(50).mean()
    d, a = float(sma.iloc[-1] - sma.iloc[-8]), atr(df)[-1]
    if np.isnan(d) or np.isnan(a):
        return None
    return "side" if abs(d) < FLAT_ATR * a else ("up" if d > 0 else "down")


def location(df, zones, last):
    """Curve analysis: price position between nearest fresh demand and fresh supply (thirds)."""
    if df is None:
        return "na"
    dem = [z["prox"] for z in zones if z["kind"] == "demand" and z["touches"] == 0 and z["prox"] <= last]
    sup = [z["prox"] for z in zones if z["kind"] == "supply" and z["touches"] == 0 and z["prox"] >= last]
    if not dem or not sup:
        return "eq"                          # notes: missing fresh side = assume equilibrium
    d, s = max(dem), min(sup)
    if s <= d:
        return "eq"
    p = (last - d) / (s - d)
    return "low" if p < 1 / 3 else "high" if p > 2 / 3 else "eq"


def rule(kind, loc, trend):
    want_loc, want_tr = ("low", "up") if kind == "demand" else ("high", "down")
    if loc == want_loc:
        return True
    if loc in ("eq", "na"):
        return None if trend is None else trend == want_tr
    return False


def grade(s):
    return next(g for m, g in GRADES if s >= m)


def analyse(meta, f):
    Z = {tf: find_zones(df, WMAX[tf]) for tf, df in f.items()}
    res = []
    for k, tf in enumerate(TFS):
        if tf not in f:
            continue
        df = f[tf]
        n, last = len(df), float(df.Close.values[-1])
        ema = [float(df.Close.ewm(span=s, adjust=False).mean().iloc[-1]) for s in (20, 50)]
        higher = [x for x in TFS[k + 1:] if x in f]
        itf, htf = MAP[tf]
        trend = trend_of(f.get(itf)) if itf else None
        loc = location(f.get(htf), Z.get(htf, []), last) if htf else "na"
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
                tail = df.tail(60)
                res.append(dict(
                    sym=meta[0], name=meta[1], tf=tf, kind=kind, pat=z["pat"], grade=grade(score), score=score,
                    parts=[fresh, z["strength"], tbase, 1 if ema_ok else 0],
                    etype="Type 1" if score >= 7 else "Type 2/3" if score >= 5 else "No trade",
                    price=round(last, 2), prox=round(z["prox"], 2), dist=round(z["dist"], 2),
                    entry=round(entry, 2), sl=round(sl, 2), target=round(float(tgt), 2),
                    nxt=None if nxt is None else round(float(nxt), 2), room=bool(room),
                    higher=higher, conf=conf, htf=htf, itf=itf, loc=loc, trend=trend,
                    rule=rule(kind, loc, trend), cred=bool(z["cred"]), nb=z["nb"], gap=z["gap"],
                    touches=z["touches"], age=n - 1 - z["i"], dist_pct=round(d, 2),
                    formed=df.index[z["i"]].strftime(fmt), idx=z["i"] - (n - 60),
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
            out += analyse(meta[t], build(daily.get(t), h60.get(t), m5.get(t)))
            ok += 1
        except Exception as e:
            print("skip", t, e)
    print(f"{len(out)} setups from {ok} stocks")
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zone Scanner</title><style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#6b7280;--card:#f4f5f7;--bd:#e2e4e8;--g:#16a34a;--r:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;--card:#1b2027;--bd:#2c333c}}
body{margin:0;padding:12px;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 2px}.mut{color:var(--mut);font-size:12px}
button,select,input{font:inherit;padding:7px 10px;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:8px}
input{width:110px}button.on{background:var(--fg);color:var(--bg)}.bar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px;cursor:pointer}
.row{display:flex;justify-content:space-between;margin-top:4px}.b{font-weight:600}
.demand{color:var(--g)}.supply{color:var(--r)}.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.tag{font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--bd);color:var(--mut)}.tag.on{color:var(--g);border-color:var(--g)}.tag.off{color:var(--r);border-color:var(--r)}
#m{position:fixed;inset:0;background:#0008;display:none;align-items:center;justify-content:center;padding:10px;overflow:auto}
#m>div{background:var(--bg);border-radius:12px;padding:12px;max-width:680px;width:100%}
rect.demand{fill:#16a34a33}rect.supply{fill:#dc262633}svg{width:100%;height:auto}
</style></head><body>
<h1>Supply &amp; Demand Scanner · Nifty 500</h1><div class="mut">Last scan: __TIME__ IST · educational only, not investment advice</div>
<div class="mut">Score = Freshness (3/1.5/0) + Strength (2/1) + Time at base (2/1/0) + EMA bonus (1). 7+ = Type 1 entry, 5-6 = Type 2/3, below 5 = no trade.</div>
<div class="bar" id="tfs"></div>
<div class="bar"><select id="side"><option value="">Demand + Supply</option><option value="demand">Demand</option><option value="supply">Supply</option></select>
<select id="sc"><option value="5">Score 5+ (tradable)</option><option value="7">Score 7+ (Type 1)</option><option value="0">All scores</option></select>
<select id="cf"><option value="0">Any confluence</option><option value="1">1+ higher TF</option><option value="2">2+ higher TF</option></select>
<select id="rl"><option value="">Any location/trend</option><option value="1">Location+trend OK</option><option value="2">OK + room for 2:1</option></select>
<select id="px"><option value="6">Within 6%</option><option value="3">Within 3%</option><option value="1">Within 1%</option></select>
<input id="q" placeholder="Search stock"><input id="cap" type="number" value="100000" title="Capital">
<select id="rk"><option value="1">Risk 1%</option><option value="1.5">Risk 1.5%</option><option value="2">Risk 2%</option></select>
<span id="cnt" class="mut" style="align-self:center"></span></div>
<div id="grid"></div><div id="m" onclick="this.style.display='none'"><div id="mc"></div></div>
<script>
const D=__DATA__,$=id=>document.getElementById(id),
N={"60m":"60 min","75m":"75 min","125m":"125 min",D:"Daily",W:"Weekly",M:"Monthly",Q:"Quarterly"},
LC={low:"Low",eq:"Equilibrium",high:"High",na:"n/a"},TR={up:"Up",down:"Down",side:"Sideways"};let tf="D",L=[];
Object.keys(N).filter(k=>D.some(z=>z.tf==k)).forEach(k=>{const b=document.createElement("button");b.textContent=N[k];if(k==tf)b.classList.add("on");
b.onclick=()=>{tf=k;[...$("tfs").children].forEach(x=>x.classList.toggle("on",x==b));draw()};$("tfs").appendChild(b)});
const t3=(v,a,b)=>v===true?`<span class="tag on">${a} ✓</span>`:v===false?`<span class="tag off">${b||a} ✗</span>`:`<span class="tag">${a} –</span>`;
const qty=z=>Math.floor((+$("cap").value||0)*(+$("rk").value)/100/Math.abs(z.entry-z.sl));
const chips=z=>z.higher.length?z.higher.map(x=>`<span class="tag ${z.conf.includes(x)?"on":""}">${x} ${z.conf.includes(x)?"✓":"✗"}</span>`).join(""):'<span class="tag">Highest TF</span>';
function draw(){const s=$("side").value,g=+$("sc").value,q=$("q").value.toLowerCase(),px=+$("px").value,cf=+$("cf").value,rl=$("rl").value;
L=D.filter(z=>z.tf==tf&&(!s||z.kind==s)&&z.score>=g&&z.conf.length>=cf&&Math.abs(z.dist_pct)<=px&&(!rl||(z.rule===true&&(rl=="1"||z.room)))&&(z.sym+z.name).toLowerCase().includes(q))
.sort((a,b)=>b.score-a.score||b.conf.length-a.conf.length||Math.abs(a.dist_pct)-Math.abs(b.dist_pct));
$("cnt").textContent=L.length+" setups";
$("grid").innerHTML=L.map((z,i)=>`<div class="card" onclick="chart(${i})"><div class="row"><span class="b">${z.sym}</span><span class="b ${z.kind}">${z.kind.toUpperCase()} ${z.pat} · ${z.grade}</span></div>
<div class="mut">${z.name} · score ${z.score} · ${z.etype}</div><div class="row"><span>Entry ${z.entry}</span><span>SL ${z.sl}</span></div><div class="row"><span>Target 2:1 ${z.target}</span><span>Qty ${qty(z)}</span></div>
<div class="chips"><span class="tag">Loc ${LC[z.loc]}</span><span class="tag">Trend ${z.trend?TR[z.trend]:"–"}</span>${t3(z.rule,"Rule")}${t3(z.room,"Room")}${t3(z.cred,"Credible")}</div>
<div class="chips">${chips(z)}<span class="tag">${z.touches?"Tested "+z.touches:"Fresh"}</span><span class="tag">${z.dist_pct}% away</span></div></div>`).join("")}
function chart(i){const z=L[i],c=z.candles,W=640,H=280,n=c.length,w=W/n;
const lo=Math.min(...c.map(x=>x[2]),z.sl)*.995,hi=Math.max(...c.map(x=>x[1]),z.target)*1.005,y=v=>H-(v-lo)/(hi-lo)*H;
const x0=Math.max(0,z.idx)*w;let s=`<svg viewBox="0 0 ${W} ${H}"><rect class="${z.kind}" x="${x0}" y="${y(Math.max(z.prox,z.dist))}" width="${W-x0}" height="${Math.abs(y(z.prox)-y(z.dist))}"/>`;
c.forEach((k,j)=>{const col=k[3]>=k[0]?"#16a34a":"#dc2626",x=j*w+w/2;
s+=`<line x1="${x}" x2="${x}" y1="${y(k[1])}" y2="${y(k[2])}" stroke="${col}"/><rect x="${x-w*.35}" y="${y(Math.max(k[0],k[3]))}" width="${w*.7}" height="${Math.max(1,Math.abs(y(k[0])-y(k[3])))}" fill="${col}"/>`});
[[z.entry,"#2563eb","Entry"],[z.sl,"#dc2626","SL"],[z.target,"#16a34a","Target"]].forEach(([v,col,t])=>s+=`<line x1="0" x2="${W}" y1="${y(v)}" y2="${y(v)}" stroke="${col}" stroke-dasharray="4"/><text x="4" y="${y(v)-3}" font-size="11" fill="${col}">${t} ${v}</text>`);
const p=z.parts;$("mc").innerHTML=`<div class="b">${z.sym} · ${N[z.tf]} · ${z.kind} ${z.pat} · grade ${z.grade}</div><div class="mut">Formed ${z.formed} · price ${z.price} · base candles ${z.nb}${z.gap?" · gap":""}</div>
<div class="mut">Score ${z.score} = Fresh ${p[0]} + Strength ${p[1]} + Base ${p[2]} + EMA ${p[3]} → ${z.etype}</div>
<div class="mut">Location (${z.htf?N[z.htf]:"n/a"}): ${LC[z.loc]} · Trend (${z.itf?N[z.itf]:"n/a"}): ${z.trend?TR[z.trend]:"–"}</div>
<div class="mut">Confirmed by: ${z.conf.map(x=>N[x]).join(", ")||"none"} · Opposite zone: ${z.nxt??"none"} · ${z.cred?"Credible":"Reaction of an earlier zone (non-reliable)"}</div>${s}</svg>`;$("m").style.display="flex"}
["side","sc","cf","rl","px","q","cap","rk"].forEach(id=>$(id).oninput=draw);draw();
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
