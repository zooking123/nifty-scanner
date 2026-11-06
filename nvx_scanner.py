#!/usr/bin/env python3
"""Supply & Demand Zone Scanner for Nifty 500 (educational use, not advice).

Setup:  pip install yfinance pandas numpy
Run:    python nvx_scanner.py              (one scan -> nvxscan.html)
        python nvx_scanner.py --watch 30   (re-scan every 30 min, page auto-reloads)
"""
import argparse, io, json, time, urllib.request, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
MAX_DIST = 6.0      # show zones within 6% of price
GRADES = [(9, "A+"), (7, "A"), (5, "B"), (0, "C")]


def symbols():
    try:
        req = urllib.request.Request(LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=20).read()
        df = pd.read_csv(io.BytesIO(raw))
        df.to_csv("nifty500list.csv", index=False)
    except Exception as e:
        print("NSE list download failed, using local nifty500list.csv:", e)
        df = pd.read_csv("nifty500list.csv")
    return {s + ".NS": (s, n) for s, n in zip(df["Symbol"], df["Company Name"])}


def fetch(tickers):
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        print(f"Downloading {i + 1}-{i + len(chunk)} of {len(tickers)}")
        d = yf.download(chunk, period="3y", interval="1d", group_by="ticker",
                        auto_adjust=True, progress=False, threads=True)
        for t in chunk:
            try:
                x = d[t].dropna()
                if len(x) > 250:
                    out[t] = x
            except Exception:
                pass
    return out


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def find_zones(df):
    """Base of 1-3 small candles followed by a strong impulse candle."""
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
        b = slice(j + 1, i)
        if j + 1 == i:
            continue
        up = c[i] > o[i]
        if up and c[i] <= h[b].max() or (not up) and c[i] >= l[b].min():
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
        if broken or abs(prox - dist) / prox > 0.06:
            continue
        touches = (int(e[0]) + int((e[1:] & ~e[:-1]).sum())) if len(e) else 0
        zones.append(dict(i=i, kind=kind, prox=float(prox), dist=float(dist),
                          imp=float(body[i] / a[i - 1]), nbase=i - (j + 1), touches=touches))
    return zones


def grade(s):
    return next(g for m, g in GRADES if s >= m)


def analyse(t, meta, df):
    c, h, n = df.Close.values, df.High.values, len(df)
    last = float(c[-1])
    s50, s200 = df.Close.rolling(50).mean().iloc[-1], df.Close.rolling(200).mean().iloc[-1]
    wk = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    wz = find_zones(wk)
    res = []
    for kind in ("demand", "supply"):
        cand = []
        for z in find_zones(df):
            if z["kind"] != kind:
                continue
            d = (last - z["prox"]) / last * 100 if kind == "demand" else (z["prox"] - last) / last * 100
            if d > MAX_DIST or d < -100:
                continue
            cand.append((abs(d), d, z))
        for _, d, z in sorted(cand, key=lambda x: x[0])[:2]:
            lo, hi = sorted((z["prox"], z["dist"]))
            htf = any(w["kind"] == kind and lo <= max(w["prox"], w["dist"]) and hi >= min(w["prox"], w["dist"])
                      for w in wz)
            trend = bool(last > s50 > s200) if kind == "demand" else bool(last < s50 < s200)
            entry = z["prox"]
            if kind == "demand":
                sl = z["dist"] * 0.999
                risk = entry - sl
                top = float(h[-120:].max())
                tgt = top if top > entry + 1.5 * risk else entry + 2 * risk
            else:
                sl = z["dist"] * 1.001
                risk = sl - entry
                bot = float(df.Low.values[-120:].min())
                tgt = bot if bot < entry - 1.5 * risk else entry - 2 * risk
            rr = abs(tgt - entry) / risk
            age = n - 1 - z["i"]
            score = (3 if z["touches"] == 0 else 1 if z["touches"] == 1 else 0) \
                + (2 if z["imp"] >= 2.5 else 1) + (1 if z["nbase"] <= 2 else 0) \
                + (2 if htf else 0) + (1 if trend else 0) + (1 if rr >= 3 else 0) + (1 if age <= 60 else 0)
            tail = df.tail(90)
            res.append(dict(
                sym=meta[0], name=meta[1], kind=kind, grade=grade(score), score=score,
                price=round(last, 2), prox=round(z["prox"], 2), dist=round(z["dist"], 2),
                entry=round(entry, 2), sl=round(sl, 2), target=round(float(tgt), 2), rr=round(rr, 2),
                htf=bool(htf), trend=trend, age=age, touches=z["touches"],
                dist_pct=round(d, 2), formed=df.index[z["i"]].strftime("%d %b %y"),
                idx=z["i"] - (n - 90),
                candles=[[round(float(v), 2) for v in r] for r in tail[["Open", "High", "Low", "Close"]].values]))
    return res


def scan():
    meta = symbols()
    data = fetch(list(meta))
    out = []
    for t, df in data.items():
        try:
            out += analyse(t, meta[t], df)
        except Exception as e:
            print("skip", t, e)
    print(f"{len(out)} setups from {len(data)} stocks")
    return out


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
__REFRESH__<title>NVX Zone Scanner</title><style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#6b7280;--card:#f4f5f7;--bd:#e2e4e8;--g:#16a34a;--r:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;--card:#1b2027;--bd:#2c333c}}
body{margin:0;padding:12px;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 2px}.mut{color:var(--mut);font-size:12px}
.tabs button,select,input{font:inherit;padding:7px 10px;border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:8px}
.tabs button.on{background:var(--fg);color:var(--bg)}.bar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px;cursor:pointer}
.row{display:flex;justify-content:space-between;margin-top:4px}.b{font-weight:600}
.demand{color:var(--g)}.supply{color:var(--r)}.tag{font-size:11px;padding:1px 6px;border-radius:6px;border:1px solid var(--bd)}
#m{position:fixed;inset:0;background:#0008;display:none;align-items:center;justify-content:center;padding:10px}
#m>div{background:var(--bg);border-radius:12px;padding:12px;max-width:680px;width:100%}
rect.demand{fill:#16a34a33}rect.supply{fill:#dc262633}svg{width:100%;height:auto}
</style></head><body>
<h1>NVX Zone Scanner · Nifty 500</h1><div class="mut">Last scan: __TIME__ · daily candles · educational only, not investment advice</div>
<div class="bar tabs"><button class="on">Simple Zone</button><button>Zone + HTF</button><button>Zone + HTF + Trend</button></div>
<div class="bar"><select id="side"><option value="">Demand + Supply</option><option value="demand">Demand</option><option value="supply">Supply</option></select>
<select id="grade"><option value="">Any grade</option><option>A+</option><option>A</option><option>B</option></select>
<select id="px"><option value="6">Within 6%</option><option value="3">Within 3%</option><option value="1">Within 1%</option></select>
<input id="q" placeholder="Search stock"><span id="cnt" class="mut" style="align-self:center"></span></div>
<div id="grid"></div><div id="m" onclick="this.style.display='none'"><div id="mc"></div></div>
<script>
const D=__DATA__,$=id=>document.getElementById(id),R={"A+":4,A:3,B:2,C:1};let tab=0,L=[];
function draw(){const s=$("side").value,g=R[$("grade").value]||0,q=$("q").value.toLowerCase(),px=+$("px").value;
L=D.filter(z=>(tab<1||z.htf)&&(tab<2||z.trend)&&(!s||z.kind==s)&&R[z.grade]>=g&&Math.abs(z.dist_pct)<=px&&(z.sym+z.name).toLowerCase().includes(q))
.sort((a,b)=>b.score-a.score||Math.abs(a.dist_pct)-Math.abs(b.dist_pct));
$("cnt").textContent=L.length+" setups";
$("grid").innerHTML=L.map((z,i)=>`<div class="card" onclick="chart(${i})"><div class="row"><span class="b">${z.sym}</span><span class="b ${z.kind}">${z.kind.toUpperCase()} · ${z.grade}</span></div>
<div class="mut">${z.name}</div><div class="row"><span>Entry ${z.entry}</span><span>SL ${z.sl}</span></div><div class="row"><span>Target ${z.target}</span><span>RR ${z.rr}</span></div>
<div class="row"><span class="tag">${z.htf?"HTF ✓":"HTF ✗"}</span><span class="tag">${z.trend?"Trend ✓":"Trend ✗"}</span><span class="tag">Age ${z.age}d</span><span class="tag">${z.dist_pct}% away</span></div></div>`).join("")}
function chart(i){const z=L[i],c=z.candles,W=640,H=280,n=c.length,w=W/n;
const lo=Math.min(...c.map(x=>x[2]),z.sl)*.995,hi=Math.max(...c.map(x=>x[1]),z.target)*1.005,y=v=>H-(v-lo)/(hi-lo)*H;
const x0=Math.max(0,z.idx)*w;let s=`<svg viewBox="0 0 ${W} ${H}"><rect class="${z.kind}" x="${x0}" y="${y(Math.max(z.prox,z.dist))}" width="${W-x0}" height="${Math.abs(y(z.prox)-y(z.dist))}"/>`;
c.forEach((k,j)=>{const col=k[3]>=k[0]?"#16a34a":"#dc2626",x=j*w+w/2;
s+=`<line x1="${x}" x2="${x}" y1="${y(k[1])}" y2="${y(k[2])}" stroke="${col}"/><rect x="${x-w*.35}" y="${y(Math.max(k[0],k[3]))}" width="${w*.7}" height="${Math.max(1,Math.abs(y(k[0])-y(k[3])))}" fill="${col}"/>`});
[[z.entry,"#2563eb","Entry"],[z.sl,"#dc2626","SL"],[z.target,"#16a34a","Target"]].forEach(([v,col,t])=>s+=`<line x1="0" x2="${W}" y1="${y(v)}" y2="${y(v)}" stroke="${col}" stroke-dasharray="4"/><text x="4" y="${y(v)-3}" font-size="11" fill="${col}">${t} ${v}</text>`);
$("mc").innerHTML=`<div class="b">${z.sym} · ${z.kind} · grade ${z.grade} (score ${z.score})</div><div class="mut">Formed ${z.formed} · touches ${z.touches} · price ${z.price}</div>${s}</svg>`;$("m").style.display="flex"}
document.querySelectorAll(".tabs button").forEach((b,i)=>b.onclick=()=>{tab=i;document.querySelectorAll(".tabs button").forEach((x,j)=>x.classList.toggle("on",i==j));draw()});
["side","grade","px","q"].forEach(id=>$(id).oninput=draw);draw();
</script></body></html>"""


def write_html(data, path, refresh):
    html = (PAGE.replace("__DATA__", json.dumps(data)).replace("__TIME__", dt.datetime.now().strftime("%d %b %Y %H:%M"))
            .replace("__REFRESH__", f'<meta http-equiv="refresh" content="{refresh * 60}">' if refresh else ""))
    open(path, "w", encoding="utf-8").write(html)
    print("Saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="re-scan every N minutes")
    ap.add_argument("--out", default="nvxscan.html")
    a = ap.parse_args()
    while True:
        write_html(scan(), a.out, a.watch)
        if not a.watch:
            break
        time.sleep(a.watch * 60)
