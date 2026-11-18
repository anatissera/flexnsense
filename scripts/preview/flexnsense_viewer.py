"""Standalone visualizer for the HF dataset JustoDario/flexnsense_skill1.

Single file: a tiny HTTP server (stdlib + pyarrow) that serves one HTML page.
Episodes are downloaded on demand with HF_TOKEN (environment or .env) and cached
under recordings/flexnsense_cache (git-ignored).

    .venv/bin/python scripts/preview/flexnsense_viewer.py            # http://127.0.0.1:8765
    .venv/bin/python scripts/preview/flexnsense_viewer.py --port 9000 --no-browser

Camera images are served straight from the parquet bytes; nothing is re-encoded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import urllib.request
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow.parquet as pq

REPO = "JustoDario/flexnsense_skill1"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "recordings" / "flexnsense_cache"
EP_RE = re.compile(r"^[\w\-./]+episode_\d+\.parquet$")


def get_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*(.*)", line)
            if m:
                return m.group(1).strip().strip("'\"")
    return None


def hf_request(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url)
    tok = get_token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    return req


_episodes_cache: list[dict] | None = None


def list_episodes() -> list[dict]:
    global _episodes_cache
    if _episodes_cache is None:
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
        with urllib.request.urlopen(hf_request(url), timeout=30) as r:
            tree = json.load(r)
        _episodes_cache = sorted(
            ({"path": t["path"], "size": t["size"]} for t in tree
             if t["type"] == "file" and t["path"].endswith(".parquet")),
            key=lambda e: e["path"],
        )
    return _episodes_cache


def local_path(path: str) -> Path:
    return CACHE / path


_dl_lock = threading.Lock()


def ensure_downloaded(path: str) -> Path:
    if not EP_RE.match(path) or ".." in path:
        raise ValueError("bad episode path")
    dst = local_path(path)
    with _dl_lock:
        if dst.exists():
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(hf_request(url), timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dst)
    return dst


class Episode:
    def __init__(self, path: str):
        table = pq.read_table(ensure_downloaded(path))
        self.cols = table.to_pydict()
        self.n = table.num_rows
        self.images = self.cols["images"]
        joints = self.cols["joints"]
        self.n_joints = max((len(j) for j in joints), default=0)
        keys = ["position_norm", "goal_norm", "position", "goal", "current_ma", "velocity"]
        series = {k: [[None] * self.n for _ in range(self.n_joints)] for k in keys}
        for i, frame in enumerate(joints):
            for j, jd in enumerate(frame):
                for k in keys:
                    series[k][j][i] = jd[k]
        ranges = []
        for j in range(self.n_joints):
            jd = joints[0][j] if joints and len(joints[0]) > j else {}
            ranges.append([jd.get("range_min"), jd.get("range_max")])
        t = self.cols["timestamp_ns_since_episode_start"]
        self.meta = {
            "path": path,
            "frames": self.n,
            "task": (self.cols["task"] or [""])[0],
            "t_ms": [x / 1e6 for x in t],
            "n_cams": max((len(im) for im in self.images), default=0),
            "n_joints": self.n_joints,
            "ranges": ranges,
            "series": series,
        }


_ep_cache: "OrderedDict[str, Episode]" = OrderedDict()
_ep_lock = threading.Lock()


def get_episode(path: str) -> Episode:
    with _ep_lock:
        if path in _ep_cache:
            _ep_cache.move_to_end(path)
            return _ep_cache[path]
    ep = Episode(path)
    with _ep_lock:
        _ep_cache[path] = ep
        while len(_ep_cache) > 3:
            _ep_cache.popitem(last=False)
    return ep


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/episodes":
                eps = list_episodes()
                for e in eps:
                    e["cached"] = local_path(e["path"]).exists()
                self.json({"repo": REPO, "episodes": eps})
            elif u.path == "/api/episode":
                self.json(get_episode(q["path"][0]).meta)
            elif u.path == "/frame":
                ep = get_episode(q["path"][0])
                i, c = int(q["i"][0]), int(q["cam"][0])
                self.send(200, ep.images[i][c]["jpeg"], "image/jpeg", cache=True)
            else:
                self.send(404, b"not found", "text/plain")
        except (KeyError, IndexError, ValueError) as e:
            self.json({"error": f"bad request: {e}"}, 400)
        except Exception as e:  # network, auth, parquet errors
            self.json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not get_token():
        print("warning: no HF_TOKEN found (environment or .env); gated dataset will fail")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"flexnsense viewer on {url}  (cache: {CACHE})")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flexnsense Viewer</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--line:#2a2f3a;--fg:#e6e8ee;--mut:#8b93a5;--acc:#5ea1ff}
@media (prefers-color-scheme: light){:root{--bg:#f4f5f8;--panel:#fff;--line:#dcdfe6;--fg:#1c2030;--mut:#5d667a;--acc:#1d62d8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:16px;margin:0 8px 0 0}
select,button{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}
main{padding:16px;display:grid;gap:16px;max-width:1500px;margin:auto}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.cam{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cam div{padding:4px 10px;color:var(--mut);font-size:12px}
.cam img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.ctrl{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
input[type=range]{flex:1;min-width:200px;accent-color:var(--acc)}
.mut{color:var(--mut)} .mono{font-variant-numeric:tabular-nums}
.joints{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:8px}
.joints label{display:flex;gap:5px;align-items:center;cursor:pointer;font-size:13px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
canvas{width:100%;height:280px;display:block;cursor:crosshair}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:3px 8px;text-align:right;border-bottom:1px solid var(--line)} th:first-child,td:first-child{text-align:left}
.bar{height:8px;background:var(--line);border-radius:4px;min-width:90px;position:relative} .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.bar b{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--fg)}
#msg{color:#e5645b}
</style></head><body>
<header>
  <h1>Flexnsense Viewer</h1>
  <select id="ep"></select>
  <span id="task" class="mut"></span>
  <span id="msg"></span>
</header>
<main>
  <div class="cams" id="cams"></div>
  <div class="card">
    <div class="ctrl">
      <button id="play">Play</button>
      <input type="range" id="scrub" min="0" max="0" value="0">
      <span class="mono" id="pos">0 / 0</span>
      <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>x
    </div>
  </div>
  <div class="card">
    <div class="ctrl"><b>Joint signal</b>
      <select id="metric">
        <option value="position_norm">position (normalized)</option>
        <option value="goal_norm">goal (normalized)</option>
        <option value="current_ma">current (mA)</option>
        <option value="velocity">velocity</option>
        <option value="position">position (raw ticks)</option>
      </select>
      <label class="mut"><input type="checkbox" id="goalov" checked> overlay goal (dashed, for position)</label>
    </div>
    <div class="joints" id="joints"></div>
    <canvas id="chart"></canvas>
  </div>
  <div class="card"><table id="tbl"></table></div>
</main>
<script>
const $=id=>document.getElementById(id);
const COLORS=['#5ea1ff','#ff7a59','#4cc38a','#f2c14e','#b08cff','#ff6fa8','#3fd0d4','#a3b0c2'];
let E=null, idx=0, playing=false, tPlay=0, raf=0, active=[];

async function api(u){const r=await fetch(u);const j=await r.json();if(j.error)throw new Error(j.error);return j}
function msg(t){$('msg').textContent=t||''}

async function init(){
  try{
    const d=await api('/api/episodes');
    $('ep').innerHTML=d.episodes.map(e=>`<option value="${e.path}">${e.path.split('/').pop().replace('.parquet','')} (${(e.size/1e6).toFixed(1)} MB${e.cached?', cached':''})</option>`).join('');
    $('ep').onchange=()=>load($('ep').value);
    if(d.episodes.length) load(d.episodes[0].path);
  }catch(e){msg(e.message)}
}

async function load(path){
  stop(); msg('Loading episode (first load downloads it)...'); $('ep').disabled=true;
  try{
    E=await api('/api/episode?path='+encodeURIComponent(path));
    E.path=path; active=E.series.position_norm.map((_,i)=>true);
    $('task').textContent=E.task+' - '+E.frames+' frames, '+(E.t_ms[E.frames-1]/1000).toFixed(1)+' s';
    $('cams').innerHTML=Array.from({length:E.n_cams},(_,c)=>`<div class="cam"><div>camera ${c}</div><img id="cam${c}" alt=""></div>`).join('');
    $('joints').innerHTML=active.map((_,j)=>`<label><input type="checkbox" data-j="${j}" checked><span class="sw" style="background:${COLORS[j%8]}"></span>joint ${j}</label>`).join('');
    $('joints').querySelectorAll('input').forEach(cb=>cb.onchange=()=>{active[cb.dataset.j]=cb.checked;draw()});
    $('scrub').max=E.frames-1; setIdx(0); msg('');
  }catch(e){msg(e.message)}
  $('ep').disabled=false;
}

function setIdx(i){
  idx=Math.max(0,Math.min(E.frames-1,i|0)); $('scrub').value=idx;
  $('pos').textContent=`${idx+1} / ${E.frames}  ${(E.t_ms[idx]/1000).toFixed(2)} s`;
  for(let c=0;c<E.n_cams;c++){const im=$('cam'+c); if(im) im.src=`/frame?path=${encodeURIComponent(E.path)}&i=${idx}&cam=${c}`}
  table(); draw();
}

function table(){
  const S=E.series;
  let h='<tr><th>joint</th><th>position</th><th>range</th><th>pos norm</th><th>goal norm</th><th>current mA</th><th>velocity</th></tr>';
  for(let j=0;j<E.n_joints;j++){
    const p=S.position_norm[j][idx],g=S.goal_norm[j][idx];
    h+=`<tr><td><span class="sw" style="background:${COLORS[j%8]}"></span> joint ${j}</td><td>${S.position[j][idx]}</td><td>${E.ranges[j][0]}-${E.ranges[j][1]}</td>
    <td><div class="bar"><i style="width:${p*100}%;background:${COLORS[j%8]}"></i><b style="left:${g*100}%"></b></div></td>
    <td>${g.toFixed(3)}</td><td>${S.current_ma[j][idx]}</td><td>${S.velocity[j][idx]}</td></tr>`;
  }
  $('tbl').innerHTML=h;
}

function draw(){
  if(!E) return;
  const cv=$('chart'),dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
  cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);
  const css=getComputedStyle(document.documentElement),mut=css.getPropertyValue('--mut'),line=css.getPropertyValue('--line'),fg=css.getPropertyValue('--fg');
  const m=$('metric').value, ov=$('goalov').checked&&m==='position_norm';
  const sets=[];
  E.series[m].forEach((s,j)=>{if(active[j])sets.push({j,s,dash:false})});
  if(ov)E.series.goal_norm.forEach((s,j)=>{if(active[j])sets.push({j,s,dash:true})});
  let lo=Infinity,hi=-Infinity;
  sets.forEach(o=>o.s.forEach(v=>{if(v<lo)lo=v;if(v>hi)hi=v}));
  if(!isFinite(lo)){lo=0;hi=1} if(lo===hi){lo-=1;hi+=1}
  const pad=(hi-lo)*0.05;lo-=pad;hi+=pad;
  const L=52,R=8,T=8,B=22,t1=E.t_ms[E.frames-1]||1;
  const X=t=>L+(W-L-R)*t/t1, Y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  g.font='11px system-ui';g.fillStyle=mut;g.strokeStyle=line;g.lineWidth=1;
  for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=Y(v);g.beginPath();g.moveTo(L,y);g.lineTo(W-R,y);g.stroke();g.textAlign='right';g.fillText(Math.abs(v)>=100?v.toFixed(0):v.toFixed(2),L-6,y+4)}
  g.textAlign='center';for(let k=0;k<=5;k++){const t=t1*k/5;g.fillText((t/1000).toFixed(1)+'s',X(t),H-6)}
  sets.forEach(o=>{g.strokeStyle=COLORS[o.j%8];g.lineWidth=1.5;g.globalAlpha=o.dash?0.6:1;g.setLineDash(o.dash?[4,3]:[]);g.beginPath();
    const step=Math.max(1,Math.floor(E.frames/(W*2)));
    for(let i=0;i<E.frames;i+=step){const x=X(E.t_ms[i]),y=Y(o.s[i]);i?g.lineTo(x,y):g.moveTo(x,y)}g.stroke()});
  g.setLineDash([]);g.globalAlpha=1;g.strokeStyle=fg;g.lineWidth=1;const cx=X(E.t_ms[idx]);g.beginPath();g.moveTo(cx,T);g.lineTo(cx,H-B);g.stroke();
}

function seekFromChart(ev){
  const r=$('chart').getBoundingClientRect(),L=52,W=r.width;
  const t=(ev.clientX-r.left-L)/(W-L-8)*E.t_ms[E.frames-1];
  let lo=0,hi=E.frames-1;while(lo<hi){const mid=(lo+hi)>>1;E.t_ms[mid]<t?lo=mid+1:hi=mid}
  setIdx(lo);
}
let drag=false;
$('chart').onmousedown=e=>{drag=true;stop();seekFromChart(e)};
addEventListener('mousemove',e=>{if(drag)seekFromChart(e)});addEventListener('mouseup',()=>drag=false);

function stop(){playing=false;cancelAnimationFrame(raf);$('play').textContent='Play'}
function start(){
  if(!E)return; if(idx>=E.frames-1)setIdx(0);
  playing=true;$('play').textContent='Pause';
  const t0=performance.now(),m0=E.t_ms[idx],sp=parseFloat($('speed').value);
  const tick=now=>{
    if(!playing)return;
    const target=m0+(now-t0)*sp;let i=idx;
    while(i<E.frames-1&&E.t_ms[i+1]<=target)i++;
    if(i!==idx)setIdx(i);
    if(i>=E.frames-1){stop();return}
    raf=requestAnimationFrame(tick);
  };
  raf=requestAnimationFrame(tick);
}
$('play').onclick=()=>playing?stop():start();
$('speed').onchange=()=>{if(playing){stop();start()}};
$('scrub').oninput=e=>{stop();setIdx(+e.target.value)};
$('metric').onchange=draw;$('goalov').onchange=draw;addEventListener('resize',draw);
addEventListener('keydown',e=>{
  if(!E||e.target.tagName==='SELECT')return;
  if(e.code==='Space'){e.preventDefault();playing?stop():start()}
  if(e.code==='ArrowRight'){stop();setIdx(idx+(e.shiftKey?10:1))}
  if(e.code==='ArrowLeft'){stop();setIdx(idx-(e.shiftKey?10:1))}
});
init();
</script></body></html>
"""

if __name__ == "__main__":
    main()
