"""Date-aware features and auditable splits; all output paths are under ROOT."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';TAB=ROOT/'results/tables';FIG=ROOT/'results/figures';MODELS=ROOT/'results/models'
RNG=42;BLOCK_M=500;TARGET_COL='D_20211214';CUTOFF='D_20201101'

def load_points():
    d=pd.read_csv(DATA/'points_base.csv');return d,sorted(c for c in d if c.startswith('D_'))

def metric_xy(d):
    x,y=Transformer.from_crs(4326,32651,always_xy=True).transform(d.lon.to_numpy(),d.lat.to_numpy())
    return np.column_stack([x,y])

def spatial_splits(d):
    xy=metric_xy(d);b=np.floor(xy/BLOCK_M).astype(int);groups=np.array([f'{x}_{y}' for x,y in b])
    u,n=np.unique(groups,return_counts=True);r=np.random.default_rng(RNG)
    order=np.lexsort((r.random(len(u)),-n));loads=np.zeros(5,int);assign={}
    for i in order:
        k=int(np.argmin(loads));assign[u[i]]=k;loads[k]+=n[i]
    sf=np.array([assign[g] for g in groups]);rf=np.empty(len(d),int)
    ix=r.permutation(len(d));rf[ix]=np.arange(len(d))%5
    return pd.DataFrame({'FID':d.FID,'lon':d.lon,'lat':d.lat,'easting':xy[:,0],'northing':xy[:,1],
                         'block':groups,'spatial_fold':sf,'random_fold':rf})

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y
    return {'R2':float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),'RMSE':float(np.sqrt(np.mean(e*e))),
            'MAE':float(np.mean(abs(e))),'bias':float(e.mean()),'n':len(y)}

def climate_at(origin,d):
    raw=pd.read_csv(DATA/'climate_daily.csv',parse_dates=['date'])
    meta=json.loads((DATA/'climate_metadata.json').read_text(encoding='utf-8'));z=float(meta['elevation'])
    origin=pd.Timestamp(origin);hist=raw[(raw.date<=origin)&(raw.date>origin-pd.Timedelta(days=365))]
    assert len(hist)>=300
    temp=hist.temperature_2m_mean.to_numpy()[:,None]-.0065*(d.elev.to_numpy()[None,:]-z)
    return pd.DataFrame({'thaw_index_365':np.clip(temp,0,None).sum(0),'freeze_index_365':np.clip(-temp,0,None).sum(0),
      'temp_30':temp[-30:].mean(0),'precip_90':np.repeat(hist.precipitation_sum.tail(90).sum(),len(d)),
      'snow_days_365':np.repeat((hist.snow_depth_max>=.02).sum(),len(d))},index=d.index)

def features_at(d,dates,origin,target,include_climate=True):
    pre=[c for c in dates if c[2:]<=pd.Timestamp(origin).strftime('%Y%m%d')];assert len(pre)>=4
    t=pd.to_datetime([c[2:] for c in pre]);tt=(t-t[0]).days.to_numpy()/365.25;Y=d[pre].to_numpy()
    A=np.column_stack([np.ones(len(tt)),tt,np.sin(2*np.pi*tt),np.cos(2*np.pi*tt)])
    coef=np.linalg.solve(A.T@A+np.diag([0,0,1,1]),A.T@Y.T);season=A[-1,2:]@coef[2:]
    dt=(pd.Timestamp(target)-pd.Timestamp(origin)).days;aspect=np.deg2rad(d.aspect.fillna(0).to_numpy())
    out=pd.DataFrame({'elev':d.elev,'slope':d.slope,'rough':d.rough,'relief':d.relief,
      'aspect_sin':np.sin(aspect),'aspect_cos':np.cos(aspect),'d_last':Y[:,-1],'trend':coef[1],
      'seasonal_amplitude':np.hypot(coef[2],coef[3]),'deseasonalized_last':Y[:,-1]-season,
      'recent_trend':np.polyfit(tt[-4:],Y[:,-4:].T,1)[0],'last_change':Y[:,-1]-Y[:,-2],
      'history_days':np.repeat((t[-1]-t[0]).days,len(d)),'horizon_days':np.repeat(dt,len(d)),
      'origin_sin':np.repeat(np.sin(2*np.pi*pd.Timestamp(origin).dayofyear/365.25),len(d)),
      'origin_cos':np.repeat(np.cos(2*np.pi*pd.Timestamp(origin).dayofyear/365.25),len(d))},index=d.index)
    if include_climate:out=pd.concat([out,climate_at(origin,d)],axis=1)
    hist=Y[:,-8:];lags=(t[-hist.shape[1]:]-t[-1]).days.to_numpy()/365.25
    seq=np.stack([hist-Y[:,-1,None],np.broadcast_to(lags,hist.shape)],axis=-1)
    if seq.shape[1]<8:seq=np.pad(seq,((0,0),(8-seq.shape[1],0),(0,0)),mode='edge')
    return out,seq,Y[:,-1],coef

def temporal_panel(d,dates,horizon=90):
    frames=[];seqs=[];meta=[]
    for target in dates:
        td=pd.Timestamp(target[2:]);eligible=[c for c in dates if pd.Timestamp(c[2:])<=td-pd.Timedelta(days=horizon)]
        if len(eligible)<4:continue
        od=pd.Timestamp(eligible[-1][2:]);X,S,last,_=features_at(d,dates,od,td)
        frames.append(X);seqs.append(S);meta.append(pd.DataFrame({'FID':d.FID,'origin':od.strftime('%Y-%m-%d'),
          'target':td.strftime('%Y-%m-%d'),'last':last,'y':d[target].to_numpy(),'horizon_days':(td-od).days}))
    return pd.concat(frames,ignore_index=True),np.concatenate(seqs),pd.concat(meta,ignore_index=True)

def save_table(df,name):
    TAB.mkdir(parents=True,exist_ok=True);df.to_csv(TAB/name,index=False,float_format='%.10g')
