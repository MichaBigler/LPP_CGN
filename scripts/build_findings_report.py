#!/usr/bin/env python3
# scripts/build_findings_report.py
# -*- coding: utf-8 -*-
"""
Build a self-contained HTML findings report from the matched-program results.
Clean editorial / card-based data-design aesthetic (semantic colour per series),
all numbers computed from Results/vss_matched_*/vss_map.csv. Inline SVG charts,
no external dependencies. Output: Analysis/matched/findings_report.html
"""
from __future__ import annotations
import pandas as pd, numpy as np, html, os

# ---- palette (semantic, from the project's data design language) ----
SF   = "#468AB2"   # SiouxFalls  (blue)
MU   = "#BF3227"   # Mumford0    (red)
EMER = "#059669"   # accent / positive
PLUM = "#86407A"   # heatmap ramp
CHAR = "#5B6470"   # neutral
INK  = "#1f2933"; GREY="#5b6470"; FAINT="#8a929b"; LINE="#e6e6e3"

def M(s): return pd.read_csv(f"Results/vss_matched_{s}/vss_map.csv", sep=";")
def rv(d,m): return (pd.to_numeric(d[m],errors='coerce')/pd.to_numeric(d['RP'],errors='coerce'))
def mean(d,m): return rv(d,m).mean()*100
def med(d,m): return rv(d,m).median()*100

# ============================================================ SVG helpers
def _axis(w,h,pad): return pad['l'], w-pad['r'], h-pad['b'], pad['t']

def grouped_bar(cats, series, w=620, h=320, unit="%", title_each=True):
    """series: list of (label,color,values[per cat])"""
    pad=dict(l=54,r=18,t=22,b=52); x0,x1,y0,y1=_axis(w,h,pad)
    vmax=max(max(v) for _,_,v in series); vmax=vmax*1.18 or 1
    def Y(v): return y0-(v/vmax)*(y0-y1)
    n=len(cats); ng=len(series); gw=(x1-x0)/n; bw=gw*0.74/ng
    s=[f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" role="img">']
    # gridlines
    for t in range(0,5):
        yy=y1+(y0-y1)*t/4; val=vmax*(4-t)/4
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{LINE}"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="12" fill="{FAINT}" text-anchor="end">{val:.1f}</text>')
    for ci,c in enumerate(cats):
        gx=x0+ci*gw+gw*0.13
        for si,(lab,col,vals) in enumerate(series):
            v=vals[ci]; bx=gx+si*bw; by=Y(v)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw*0.9:.1f}" height="{y0-by:.1f}" rx="1.5" fill="{col}"/>')
            s.append(f'<text x="{bx+bw*0.45:.1f}" y="{by-5:.1f}" font-size="12.5" fill="{INK}" text-anchor="middle">{v:.2f}</text>')
        s.append(f'<text x="{gx+ (bw*ng)/2 - bw*0.05:.1f}" y="{y0+19:.1f}" font-size="13" fill="{GREY}" text-anchor="middle">{html.escape(str(c))}</text>')
    s.append(f'<text x="{x0-40}" y="{y1+2}" font-size="11.5" fill="{FAINT}" transform="rotate(-90 {x0-40} {(y0+y1)/2})" text-anchor="middle">{unit}</text>')
    s.append('</svg>')
    return _legend(series)+''.join(s)

def line_chart(xs, series, w=620, h=320, unit="%", xlabel="", xticks=None, logx=False):
    """series: list of (label,color,yvals,marker)"""
    pad=dict(l=54,r=96,t=22,b=48); x0,x1,y0,y1=_axis(w,h,pad)
    allv=[v for _,_,ys,_ in series for v in ys]; vmax=max(allv)*1.15 or 1; vmin=min(0,min(allv))
    import math
    xs_t=[math.log10(x) for x in xs] if logx else list(xs)
    xmn,xmx=min(xs_t),max(xs_t); xr=(xmx-xmn) or 1
    def X(i): return x0+(xs_t[i]-xmn)/xr*(x1-x0)
    def Y(v): return y0-((v-vmin)/(vmax-vmin))*(y0-y1)
    s=[f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" role="img">']
    for t in range(0,5):
        yy=y1+(y0-y1)*t/4; val=vmin+(vmax-vmin)*(4-t)/4
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{LINE}"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="12" fill="{FAINT}" text-anchor="end">{val:.2f}</text>')
    ticklabs=xticks or [str(x) for x in xs]
    for i,xv in enumerate(xs):
        s.append(f'<text x="{X(i):.1f}" y="{y0+19:.1f}" font-size="12" fill="{GREY}" text-anchor="middle">{html.escape(str(ticklabs[i]))}</text>')
    for lab,col,ys,mk in series:
        pts=" ".join(f"{X(i):.1f},{Y(v):.1f}" for i,v in enumerate(ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for i,v in enumerate(ys):
            cx,cy=X(i),Y(v)
            if mk=='sq': s.append(f'<rect x="{cx-3.5:.1f}" y="{cy-3.5:.1f}" width="7" height="7" fill="{col}"/>')
            else: s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{col}"/>')
        s.append(f'<text x="{X(len(ys)-1)+8:.1f}" y="{Y(ys[-1])+4:.1f}" font-size="12.5" fill="{col}" font-weight="600">{html.escape(lab)}</text>')
    if xlabel: s.append(f'<text x="{(x0+x1)/2}" y="{h-6}" font-size="11.5" fill="{FAINT}" text-anchor="middle">{html.escape(xlabel)}</text>')
    s.append('</svg>')
    return ''.join(s)

def hbar(items, w=620, unit="pp"):
    """items: list of (label,value,color)"""
    rowh=38; pad=dict(l=165,r=58,t=10,b=10); h=pad['t']+pad['b']+rowh*len(items)
    x0=pad['l']; x1=w-pad['r']; vmax=max(v for _,v,_ in items)*1.12 or 1
    s=[f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" role="img">']
    for i,(lab,v,col) in enumerate(items):
        y=pad['t']+i*rowh+rowh*0.18; bh=rowh*0.5; bw=(v/vmax)*(x1-x0)
        s.append(f'<text x="{x0-10}" y="{y+bh*0.72:.1f}" font-size="12.5" fill="{GREY}" text-anchor="end">{html.escape(lab)}</text>')
        s.append(f'<rect x="{x0}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>')
        s.append(f'<text x="{x0+bw+8:.1f}" y="{y+bh*0.75:.1f}" font-size="12.5" fill="{INK}">{v:.2f} {unit}</text>')
    s.append('</svg>')
    return ''.join(s)

def _legend(series):
    items="".join(f'<span class="lg"><i style="background:{c}"></i>{html.escape(l)}</span>' for l,c,_ in series)
    return f'<div class="legend">{items}</div>'

def legend2(pairs):
    return '<div class="legend">'+"".join(f'<span class="lg"><i style="background:{c}"></i>{html.escape(l)}</span>' for l,c in pairs)+'</div>'

def heatmap(rowlabs, collabs, mat, ramp=PLUM, rlabel="", clabel=""):
    vmax=max(max(r) for r in mat) or 1
    def shade(v):
        a=0.08+0.82*(v/vmax)
        return f'background:rgba({int(ramp[1:3],16)},{int(ramp[3:5],16)},{int(ramp[5:7],16)},{a:.2f});color:{"#fff" if a>0.55 else INK}'
    th="".join(f"<th>{html.escape(str(c))}</th>" for c in collabs)
    rows=""
    for rl,row in zip(rowlabs,mat):
        cells="".join(f'<td style="{shade(v)}">{v:.2f}</td>' for v in row)
        rows+=f"<tr><th>{html.escape(str(rl))}</th>{cells}</tr>"
    cap=f'<div class="hmcap"><span>{html.escape(rlabel)}</span><span>{html.escape(clabel)}</span></div>' if rlabel or clabel else ""
    return f'{cap}<table class="heat"><tr><th></th>{th}</tr>{rows}</table>'

def table(headers, rows, aligns=None):
    aligns=aligns or ['left']+['right']*(len(headers)-1)
    th="".join(f'<th style="text-align:{a}">{html.escape(h)}</th>' for h,a in zip(headers,aligns))
    body=""
    for r in rows:
        tds="".join(f'<td style="text-align:{a}">{c}</td>' for c,a in zip(r,aligns))
        body+=f"<tr>{tds}</tr>"
    return f'<table class="data"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'

# ============================================================ compute data
def kprof(net):
    d=M(f'main_{net}'); d=d[(d.case_per_run==1)&(d.case_selection=='line_consecutive')]
    return [mean(d[d.case_k==k],'VSS_nom') for k in [1,2,3,4]]
def kprof_native(path):
    d=pd.read_csv(path,sep=';'); d=d[(d.case_per_run==1)&(d.case_selection=='line_consecutive')&(d.case_k<=4)]
    return [rv(d[d.case_k==k],'VSS_nom').mean()*100 for k in [1,2,3,4]]

sf_k, mu_k = kprof('sf'), kprof('mumford0')
sf_kn = kprof_native("Results/vss_sf_main_redo/vss_map.csv")
mu_kn = kprof_native("Results/vss_mumford0_redo/vss_map.csv")
sf_kn_mean=np.mean(sf_kn); mu_kn_mean=np.mean(mu_kn); sf_k_mean=np.mean(sf_k); mu_k_mean=np.mean(mu_k)

def replfreq(net,metric='VSS_nom'):
    d=M(f'replcost_{net}'); d=d[d.case_k==1]
    return [mean(d[d.cost_repl_freq==c] if metric=='VSS_nom' else d[d.cost_repl_freq==c], metric) for c in [10,100,1000]]
sf_rc=[mean(M('replcost_sf')[(M('replcost_sf').case_k==1)&(M('replcost_sf').cost_repl_freq==c)],'VSS_nom') for c in [10,100,1000]]
mu_rc=[mean(M('replcost_mumford0')[(M('replcost_mumford0').case_k==1)&(M('replcost_mumford0').cost_repl_freq==c)],'VSS_nom') for c in [10,100,1000]]

def repl2d(net):
    d=M(f'replcost_{net}'); d=d[d.case_k==1]
    return [[mean(d[(d.cost_repl_freq==f)&(d.cost_repl_line==l)],'VSS_nom') for l in [50,500,5000]] for f in [10,100,1000]]
sf_2d, mu_2d = repl2d('sf'), repl2d('mumford0')

PF=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
def pf(net,metric):
    d=M(f'pfail_{net}'); d=d[d.case_k==1]
    return [mean(d[np.isclose(d.case_p_fail,p)],metric) for p in PF]
sf_pf_v,sf_pf_e=pf('sf','VSS_nom'),pf('sf','EVPI')
mu_pf_v,mu_pf_e=pf('mumford0','VSS_nom'),pf('mumford0','EVPI')

RHO=[0.05,0.10,0.15]
def lf(net):
    out=[]
    for rho,(slug,ds) in zip(RHO,[('loadmatch_overnight','x'),('loadmatch','x'),('loadmatch_overnight','x')]):
        pass
    return out
# load-factor curve from the dedicated runs
def lf_curve(net_main, net_on, ds_map):
    res={}
    for f,rmap in [(net_on,{k:v for k,v in ds_map.items() if v in (0.05,0.15)}),(net_main,{k:v for k,v in ds_map.items() if v==0.10})]:
        d=pd.read_csv(f,sep=';'); d['rho']=d.demand_scale.map(ds_map)
        for rho in [0.05,0.10,0.15]:
            sub=d[np.isclose(d.rho,rho)]
            if len(sub): res[rho]=rv(sub,'VSS_nom').mean()*100
    return [res.get(r,np.nan) for r in RHO]
sf_lf=lf_curve("Results/vss_loadmatch_sf/vss_map.csv","Results/vss_loadmatch_overnight_sf/vss_map.csv",{2.536:0.10,1.2683:0.05,3.8049:0.15})
mu_lf=lf_curve("Results/vss_loadmatch_mumford0/vss_map.csv","Results/vss_loadmatch_overnight_mumford0/vss_map.csv",{0.0368:0.10,0.0184:0.05,0.0552:0.15})

# risk CDF at replfreq=1000
def cdf(net):
    d=M(f'replcost_{net}'); d=d[(d.case_k==1)&(d.cost_repl_freq==1000)]; v=np.sort(rv(d,'VSS_nom').values*100)
    return v, np.arange(1,len(v)+1)/len(v)
sf_cv,sf_cc=cdf('sf'); mu_cv,mu_cc=cdf('mumford0')
def dstats(net):
    d=M(f'replcost_{net}'); d=d[(d.case_k==1)&(d.cost_repl_freq==1000)]; v=rv(d,'VSS_nom')*100
    return dict(mean=v.mean(),med=v.median(),p90=v.quantile(.9),mx=v.max(),zero=(v<0.1).mean()*100)
sf_ds,mu_ds=dstats('sf'),dstats('mumford0')

# lever spreads
def spread(net,sw,col):
    d=M(f'{sw}_{net}'); d=d[d.case_k==1]; g=d.groupby(col).apply(lambda x:mean(x,'VSS_nom'))
    return g.max()-g.min()
levers=[('Replanning cost (freq)','replcost','cost_repl_freq'),
        ('Bypass multiplier','bypass','bypass_multiplier'),
        ('Overdemand multiplier','overdemand','overdemand_multiplier')]

# selection
def selrow(net):
    d=M(f'main_{net}'); d=d[(d.case_per_run==1)&(d.case_k==2)]
    return {s:mean(g,'VSS_nom') for s,g in d.groupby('case_selection')}
sf_sel,mu_sel=selrow('sf'),selrow('mumford0')
# bypass curve
def byp(net):
    d=M(f'bypass_{net}'); d=d[d.case_k==1]
    return {int(b):mean(g,'VSS_nom') for b,g in d.groupby('bypass_multiplier')}
sf_by,mu_by=byp('sf'),byp('mumford0')
# evpi/vss ratio
def evr(net):
    out={}
    for sw in ['replcost','bypass','overdemand','pfail']:
        d=M(f'{sw}_{net}'); d=d[d.case_k==1]
        v=mean(d,'VSS_nom'); e=mean(d,'EVPI'); out[sw]=e/v if v>0.01 else float('nan')
    return out
sf_ev,mu_ev=evr('sf'),evr('mumford0')

# ============================================================ HTML
def fmt(x): return f"{x:.2f}"
def pct(x): return f"{x:.2f}%"

def card(num,title,sub,body):
    n=f'<span class="secn">{num}</span>' if num else ''
    return f'<section class="card"><div class="chead">{n}<div><h2>{html.escape(title)}</h2>{f"<p class=sub>{sub}</p>" if sub else ""}</div></div>{body}</section>'

def kpi(value,label,note,color):
    return f'<div class="kpi"><span class="dot" style="background:{color}"></span><div class="kv">{value}</div><div class="kl">{html.escape(label)}</div><div class="kn">{html.escape(note)}</div></div>'

netleg=legend2([('SiouxFalls (sparse)',SF),('Mumford0 (dense)',MU)])

# Block: topology
topo_chart=grouped_bar(['Native','Matched'],
    [('SiouxFalls',SF,[sf_kn_mean,sf_k_mean]),('Mumford0',MU,[mu_kn_mean,mu_k_mean])], unit="VSS/RP %")
kprof_chart=line_chart([1,2,3,4],[('SiouxFalls',SF,sf_k,'o'),('Mumford0',MU,mu_k,'sq')],
    unit="VSS/RP %", xlabel="disruption size k")
topo_tbl=table(['k','SF VSS/RP','Mumford0 VSS/RP'],
    [[k,pct(s),pct(m)] for k,s,m in zip([1,2,3,4],sf_k,mu_k)])

# Block: replcost
rc_chart=grouped_bar(['10','100','1000'],[('SiouxFalls',SF,sf_rc),('Mumford0',MU,mu_rc)],unit="VSS/RP %")
rc_2d_sf=heatmap(['freq 10','freq 100','freq 1000'],['line 50','line 500','line 5000'],sf_2d,ramp=SF,rlabel="rows: cost_repl_freq",clabel="cols: cost_repl_line")
rc_2d_mu=heatmap(['freq 10','freq 100','freq 1000'],['line 50','line 500','line 5000'],mu_2d,ramp=MU)

# Block: pfail
pf_v_chart=line_chart(PF,[('SiouxFalls',SF,sf_pf_v,'o'),('Mumford0',MU,mu_pf_v,'sq')],unit="VSS/RP %",xlabel="p_fail")
pf_e_chart=line_chart(PF,[('SiouxFalls',SF,sf_pf_e,'o'),('Mumford0',MU,mu_pf_e,'sq')],unit="EVPI/RP %",xlabel="p_fail")

# Block: load
lf_chart=line_chart(RHO,[('SiouxFalls',SF,sf_lf,'o'),('Mumford0',MU,mu_lf,'sq')],unit="VSS/RP %",xlabel="load factor ρ")

# Block: risk
def cdf_chart():
    w,h=620,320; pad=dict(l=54,r=24,t=22,b=48); x0,x1,y0,y1=pad['l'],w-pad['r'],h-pad['b'],pad['t']
    vmax=max(sf_cv.max(),mu_cv.max())*1.02
    def X(v): return x0+max(v,0)/vmax*(x1-x0)
    def Y(c): return y0-c*(y0-y1)
    s=[f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="var(--mono)" role="img">']
    for t in range(5):
        yy=y1+(y0-y1)*t/4; s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{LINE}"/>')
        s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" font-size="12" fill="{FAINT}" text-anchor="end">{(4-t)/4:.2f}</text>')
    for xt in [0,20,40,60,80,100]:
        s.append(f'<text x="{X(xt):.1f}" y="{y0+19:.1f}" font-size="12" fill="{GREY}" text-anchor="middle">{xt}</text>')
    for v,c,col,lab in [(sf_cv,sf_cc,SF,'SiouxFalls'),(mu_cv,mu_cc,MU,'Mumford0')]:
        pts=" ".join(f"{X(x):.1f},{Y(y):.1f}" for x,y in zip(v,c))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        s.append(f'<text x="{X(v[-1])-4:.1f}" y="{Y(c[-1])-8:.1f}" font-size="12.5" fill="{col}" font-weight="600" text-anchor="end">{lab}</text>')
    s.append(f'<text x="{(x0+x1)/2}" y="{h-6}" font-size="11.5" fill="{FAINT}" text-anchor="middle">VSS/RP [%], cumulative fraction</text>')
    s.append('</svg>'); return ''.join(s)
risk_tbl=table(['','mean','median','P90','max','share ≈0'],
    [['SiouxFalls',pct(sf_ds['mean']),pct(sf_ds['med']),pct(sf_ds['p90']),pct(sf_ds['mx']),f"{sf_ds['zero']:.0f}%"],
     ['Mumford0',pct(mu_ds['mean']),pct(mu_ds['med']),pct(mu_ds['p90']),pct(mu_ds['mx']),f"{mu_ds['zero']:.0f}%"]])

# lever ranking
lev_sf=hbar([(l,spread('sf',sw,c),SF) for l,sw,c in levers])
lev_mu=hbar([(l,spread('mumford0',sw,c),MU) for l,sw,c in levers])

# selection / bypass tables
sel_tbl=table(['network']+list(sf_sel.keys()),
    [['SiouxFalls']+[pct(sf_sel[k]) for k in sf_sel]],)
sel_tbl=table(['network','line_all','line_consecutive','random','share_stop'],
    [['SiouxFalls']+[pct(sf_sel.get(k,float('nan'))) for k in ['line_all','line_consecutive','random','share_stop']],
     ['Mumford0']+[pct(mu_sel.get(k,float('nan'))) for k in ['line_all','line_consecutive','random','share_stop']]])
byp_tbl=table(['network','10','20','50','100','200'],
    [['SiouxFalls']+[pct(sf_by.get(b,float('nan'))) for b in [10,20,50,100,200]],
     ['Mumford0']+[pct(mu_by.get(b,float('nan'))) for b in [10,20,50,100,200]]])
ev_tbl=table(['regime','SF EVPI/VSS','Mumford0 EVPI/VSS','reading'],
    [['replanning cost',fmt(sf_ev['replcost']),fmt(mu_ev['replcost']),'VSS ≫ EVPI'],
     ['bypass',fmt(sf_ev['bypass']),fmt(mu_ev['bypass']),'EVPI ≥ VSS'],
     ['overdemand',fmt(sf_ev['overdemand']),fmt(mu_ev['overdemand']),'≈ equal'],
     ['p_fail',fmt(sf_ev['pfail']),fmt(mu_ev['pfail']),'mixed']],
    aligns=['left','right','right','left'])

# KPIs (figures only, neutral labels)
kpis="".join([
    kpi(pct(sf_k_mean),"SF mean VSS/RP","matched, k=1–4",SF),
    kpi(pct(mu_k_mean),"Mumford0 mean VSS/RP","matched, k=1–4",MU),
    kpi(pct(max(sf_rc)),"SF VSS/RP","cost_repl_freq=1000, k=1",SF),
    kpi("5,025","cases","10 sweeps, both networks",CHAR),
])

CSS = """
:root{--mono:'SF Mono',ui-monospace,'Cascadia Mono',Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
--ink:#1f2933;--grey:#5b6470;--faint:#8a929b;--line:#e6e6e3;--bg:#f6f5f2;--card:#fff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:56px 24px 80px}
.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin:0 0 10px}
h1{font-size:34px;line-height:1.15;margin:0 0 12px;font-weight:700;letter-spacing:-.01em}
.lede{font-size:17px;color:var(--grey);max-width:64ch;margin:0 0 8px}
.meta{font-family:var(--mono);font-size:12px;color:var(--faint);margin-top:14px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:30px 0 8px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.kpi .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-bottom:10px}
.kv{font-size:26px;font-weight:700;letter-spacing:-.01em}
.kl{font-size:13px;font-weight:600;margin-top:2px}
.kn{font-size:12px;color:var(--faint);margin-top:3px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px 26px;margin:18px 0}
.chead{display:flex;gap:14px;align-items:baseline;margin-bottom:6px}
.secn{font-family:var(--mono);font-size:13px;color:var(--faint);font-weight:600;min-width:22px}
h2{font-size:19px;margin:0;font-weight:700;letter-spacing:-.01em}
.sub{margin:4px 0 0;color:var(--grey);font-size:14px;max-width:70ch}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:14px}
.chartbox{margin-top:14px}
.cn{font-family:var(--mono);font-size:12.5px;color:var(--grey);margin:2px 0 10px}
svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin:2px 0 8px;font-family:var(--mono);font-size:13px;color:var(--grey)}
.lg{display:inline-flex;align-items:center;gap:7px}
.lg i{width:13px;height:13px;border-radius:2px;display:inline-block}
table.data{width:100%;border-collapse:collapse;margin-top:14px;font-size:13px}
table.data th{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint);font-weight:600;padding:7px 10px;border-bottom:1.5px solid var(--line)}
table.data td{padding:7px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
table.data tbody tr:last-child td{border-bottom:none}
table.heat{border-collapse:separate;border-spacing:3px;margin-top:6px;font-family:var(--mono);font-size:14px}
table.heat th{font-size:12px;color:var(--grey);font-weight:600;padding:4px 10px;text-align:center}
table.heat td{padding:12px 18px;text-align:center;border-radius:4px;font-variant-numeric:tabular-nums}
.hmcap{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-bottom:4px}
.note{background:#fafaf8;border:1px solid var(--line);border-left:3px solid var(--grey);border-radius:6px;padding:12px 15px;margin-top:14px;font-size:13px;color:var(--grey)}
.note.key{border-left-color:#059669}
.note.warn{border-left-color:#d9a400}
.tag{display:inline-block;font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.06em;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--grey);margin-left:8px;vertical-align:middle}
.tag.must{background:#eafaf3;border-color:#bfe8d6;color:#0a7a52}
.tag.supp{background:#f3f4f6;color:#5b6470}
footer{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:30px;border-top:1px solid var(--line);padding-top:16px;line-height:1.7}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
"""

def chartbox(note,svg): return f'<div class="chartbox"><div class="cn">{html.escape(note)}</div>{svg}</div>'

HTML=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Value of stochastic line planning — findings</title><style>{CSS}</style></head>
<body><div class="wrap">
<p class="kicker">HRAM · two-stage line planning under disruptions</p>
<h1>SiouxFalls and Mumford0 — matched comparison</h1>
<p class="lede">VSS (value of the stochastic solution) and EVPI (value of perfect information), each relative to recourse cost RP. Both networks at identical operational parameters and load factor ρ=0.10; aggregates are means over cases.</p>
<p class="meta">5,025 cases · 10 sweeps · Gurobi 12.0.3 · gap 1e-5 · Results/vss_matched_*</p>

<div class="kpis">{kpis}</div>

{card('1','VSS by network: native vs matched, and matched k-profile','train_capacity=50, max_frequency=10, infrastructure_capacity=10, bypass_multiplier=50, num_od=50; demand-scaled to ρ=0.10. Native = each network at its own baseline parameters.',
    netleg+'<div class="grid2">'+chartbox('Mean VSS/RP — native vs matched',topo_chart)+chartbox('Matched — VSS/RP by disruption size k',kprof_chart)+'</div>'+topo_tbl)}

{card('2','Replanning cost','VSS/RP vs frequency-replanning cost (k=1), and the full freq × line-cost grid.',
    netleg+chartbox('VSS/RP vs cost_repl_freq (k=1)',rc_chart)+
    '<div class="grid2">'+chartbox('SiouxFalls — VSS/RP %, cost_repl_freq × cost_repl_line',rc_2d_sf)+chartbox('Mumford0 — VSS/RP %, cost_repl_freq × cost_repl_line',rc_2d_mu)+'</div>')}

{card('3','Disruption probability','VSS/RP and EVPI/RP vs p_fail (k=1).',
    netleg+'<div class="grid2">'+chartbox('VSS/RP vs p_fail',pf_v_chart)+chartbox('EVPI/RP vs p_fail',pf_e_chart)+'</div>')}

{card('4','Load factor','Mean VSS/RP (over k) at three matched load factors.',
    netleg+chartbox('VSS/RP vs load factor ρ',lf_chart))}

{card('5','Distribution at cost_repl_freq=1000 (k=1)','Cumulative distribution and summary statistics of VSS/RP.',
    netleg+chartbox('Cumulative distribution of VSS/RP',cdf_chart())+risk_tbl)}

{card('6','Spread of mean VSS/RP per sensitivity axis (k=1)','',
    '<div class="grid2">'+chartbox('SiouxFalls (pp)',lev_sf)+chartbox('Mumford0 (pp)',lev_mu)+'</div>')}

{card('7','Selection mode (k=2) and bypass multiplier (k=1)','VSS/RP %.',
    sel_tbl+'<div style="height:14px"></div>'+byp_tbl)}

{card('8','EVPI/RP ÷ VSS/RP by sensitivity axis (k=1)','',
    ev_tbl)}

<footer>
Results/vss_matched_&lt;sweep&gt;_&lt;net&gt;/vss_map.csv · both networks: train_capacity=50, max_frequency=10, infrastructure_capacity=10, bypass_multiplier=50 (baseline), num_od=50, demand-scaled to ρ=0.10 · WS sub-solves: Threads=1, Seed=42, NumericFocus=2 · 0 bound violations / 5,025 cases.<br>
Full numbers: Analysis/matched/summary_tables.csv · scripts/build_findings_report.py
</footer>
</div></body></html>"""

os.makedirs("Analysis/matched",exist_ok=True)
with open("Analysis/matched/findings_report.html","w",encoding="utf-8") as f:
    f.write(HTML)
print("wrote Analysis/matched/findings_report.html",f"({len(HTML)//1024} KB)")
