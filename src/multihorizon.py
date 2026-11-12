"""Phase D: causal multi-lead training with past-only temperature projections.

Expected weather is a historical climatology with decaying current anomaly,
never observed future ERA5. Degree-day features are proxies, not measured ALT.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from core import ROOT,load_points,features_at,metrics
from models import Scale
from improvement import Network,initialise

OUT=ROOT/'results/multihorizon'
VDATES=['2020-04-11','2020-06-10','2020-08-09','2020-09-02','2020-10-08','2020-11-01']
CONFIGS=[
 {'name':'MultiLead GA-PSO-BP','kind':'neural','hidden':5,'ga':True},
 {'name':'MultiLead GA-PSO-BP','kind':'neural','hidden':16,'ga':True},
 {'name':'MultiLead BP','kind':'neural','hidden':5,'ga':False},
 {'name':'MultiLead BP','kind':'neural','hidden':16,'ga':False},
 {'name':'MultiLead BP','kind':'neural','hidden':64,'ga':False},
 {'name':'MultiLead LSTM','kind':'neural','hidden':8,'ga':False,'sequence':True},
 {'name':'MultiLead LSTM','kind':'neural','hidden':24,'ga':False,'sequence':True},
 {'name':'MultiLead XGBoost','kind':'xgb','depth':2},
 {'name':'MultiLead XGBoost','kind':'xgb','depth':4},
 {'name':'MultiLead RandomForest','kind':'rf','leaf':10},
 {'name':'MultiLead RandomForest','kind':'rf','leaf':30},
]
for i,c in enumerate(CONFIGS):c['id']=f'D{i:02d}'

def save(df,name):df.to_csv(OUT/name,index=False,float_format='%.10g')
class Thermal:
    def __init__(self,d):
        self.raw=pd.read_csv(ROOT/'data/climate_daily.csv',parse_dates=['date']).set_index('date')
        self.elevation=d.elev.to_numpy();self.grid=float(json.loads((ROOT/'data/climate_metadata.json').read_text())['elevation']);self.cache={}
    def projected(self,origin):
        if origin in self.cache:return self.cache[origin]
        od=pd.Timestamp(origin);r=self.raw;past=r.loc[:od].copy();older=past[past.index.year<od.year]
        clim=older.groupby(older.index.strftime('%m-%d')).temperature_2m_mean.mean().to_dict()
        if '02-29' not in clim:clim['02-29']=clim['02-28']
        recent=past.tail(30);normal=np.array([clim[x.strftime('%m-%d')] for x in recent.index]);anomaly=float(np.mean(recent.temperature_2m_mean.to_numpy()-normal))
        hist=past.temperature_2m_mean.tail(180).to_numpy();assert len(hist)==180
        future_dates=pd.date_range(od+pd.Timedelta(days=1),periods=180)
        future=np.array([clim[x.strftime('%m-%d')] for x in future_dates])+anomaly*np.exp(-np.arange(1,181)/30)
        self.cache[origin]=(hist,future,anomaly);return self.cache[origin]
    def at(self,origin,target):
        hist,future,anom=self.projected(origin);lead=(pd.Timestamp(target)-pd.Timestamp(origin)).days;assert 1<=lead<=180
        temp=np.concatenate([hist,future[:lead]])[:,None]-.0065*(self.elevation[None,:]-self.grid)
        warm=np.clip(temp,0,None);cold=np.clip(-temp,0,None)
        return {'expected_thaw_DD':warm[180:].sum(0),'expected_freeze_DD':cold[180:].sum(0),
          'expected_sqrt_thaw180_change':np.sqrt(warm[-180:].sum(0))-np.sqrt(warm[:180].sum(0)),
          'expected_sqrt_freeze180_change':np.sqrt(cold[-180:].sum(0))-np.sqrt(cold[:180].sum(0)),
          'expected_temp_target30':temp[-30:].mean(0),'origin_temp_anomaly30':np.repeat(anom,len(self.elevation))}

def make_pair(d,dates,origin,target,thermal):
    od=pd.Timestamp(origin);td=pd.Timestamp(target);X,S,last,coef=features_at(d,dates,od,td)
    for n,v in thermal.at(origin,target).items():X[n]=v
    for harmonic in [1,2]:
        X[f'target_sin_{harmonic}']=np.sin(2*np.pi*harmonic*td.day_of_year/365.25)
        X[f'target_cos_{harmonic}']=np.cos(2*np.pi*harmonic*td.day_of_year/365.25)
    for j in range(7):X[f'lag_relative_{j}']=S[:,j,0];X[f'lag_time_{j}']=S[:,j,1]
    mt=pd.DataFrame({'FID':d.FID,'origin':origin,'target':target,'last':last,'y':d['D_'+target.replace('-','')].to_numpy(),'horizon_days':(td-od).days})
    return X,S,mt

def dataset():
    d,dates=load_points();thermal=Thermal(d);pairs=[]
    for j,oc in enumerate(dates):
        if j<3:continue
        origin=pd.Timestamp(oc[2:]);
        for tc in dates[j+1:]:
            td=pd.Timestamp(tc[2:]);lead=(td-origin).days
            if 24<=lead<=180 and td<=pd.Timestamp('2020-11-01'):pairs.append((origin.strftime('%Y-%m-%d'),td.strftime('%Y-%m-%d'),'train',0))
    for horizon in [30,60,90]:
        for tc in dates:
            td=pd.Timestamp(tc[2:]);eligible=[c for c in dates if pd.Timestamp(c[2:])<=td-pd.Timedelta(days=horizon)]
            if td.year!=2021 or len(eligible)<4:continue
            origin=pd.Timestamp(eligible[-1][2:])
            if origin<pd.Timestamp('2020-11-01'):continue
            pairs.append((origin.strftime('%Y-%m-%d'),td.strftime('%Y-%m-%d'),'test',horizon))
    # Add the exact purged 90-day validation cases even when not retained as unique latest lead.
    old=pd.read_csv(ROOT/'results/tables/temporal_pairs.csv')
    for target in VDATES:
        a=old[(old.protocol=='temporal_90d')&(old.target==target)].iloc[0]
        pairs.append((a.origin,a.target,'validation',90))
    cache={};frames=[];seq=[];meta=[]
    for origin,target,role,horizon in pairs:
        key=(origin,target)
        if key not in cache:cache[key]=make_pair(d,dates,origin,target,thermal)
        x,s,m=cache[key];mm=m.copy();mm['role']=role;mm['protocol_horizon']=horizon
        frames.append(x);seq.append(s);meta.append(mm)
    X=pd.concat(frames,ignore_index=True);S=np.concatenate(seq);mt=pd.concat(meta,ignore_index=True)
    assert np.isfinite(X.to_numpy()).all();return d,dates,X,S,mt

class Predictor:
    def predict(self,X,S):
        if self.kind!='neural':return self.est.predict(X)
        xx=torch.tensor(self.scale.x(np.asarray(X)));ss=torch.tensor(self.scale.seq(S)) if self.sequence else None;out=[]
        for net in self.nets:
            net.eval();p=[]
            with torch.no_grad():
                for j in range(0,len(X),4096):p.append(net(xx[j:j+4096],ss[j:j+4096] if ss is not None else None).numpy())
            out.append(np.concatenate(p)*self.scale.ys+self.scale.ym)
        return np.mean(out,axis=0)

def fit(c,X,y,S,Xv=None,yv=None,Sv=None,steps=None,members=1):
    t0=time.perf_counter();m=Predictor();m.kind=c['kind'];m.best_steps=[];m.sequence=c.get('sequence',False)
    if m.kind=='rf':m.est=RandomForestRegressor(n_estimators=200,min_samples_leaf=c['leaf'],max_features=.8,n_jobs=2,random_state=42).fit(X,y)
    elif m.kind=='xgb':m.est=XGBRegressor(n_estimators=400,max_depth=c['depth'],min_child_weight=20,reg_lambda=10,learning_rate=.035,subsample=.8,colsample_bytree=.8,n_jobs=2,random_state=42).fit(X,y)
    else:
        m.scale=Scale().fit(np.asarray(X),y,S);xx=torch.tensor(m.scale.x(np.asarray(X)));yy=torch.tensor(m.scale.y(y));ss=torch.tensor(m.scale.seq(S)) if m.sequence else None
        xv=torch.tensor(m.scale.x(np.asarray(Xv))) if Xv is not None else None;yv0=torch.tensor(m.scale.y(yv)) if Xv is not None else None;sv=torch.tensor(m.scale.seq(Sv)) if m.sequence and Xv is not None else None
        m.nets=[]
        for seed in range(42,42+members):
            torch.manual_seed(seed);net=Network(X.shape[1],c['hidden'],m.sequence)
            if c.get('ga'):initialise(net,xx.numpy(),yy.numpy(),seed)
            opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001);gen=torch.Generator().manual_seed(seed)
            best=np.inf;bs=20;wait=0;state=None
            for step in range(1,(steps or 600)+1):
                net.train();ix=torch.randint(len(yy),(512,),generator=gen);opt.zero_grad();loss=((net(xx[ix],ss[ix] if ss is not None else None)-yy[ix])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                if xv is not None and step%20==0:
                    net.eval()
                    with torch.no_grad():v=float(((net(xv,sv)-yv0)**2).mean())
                    if v<best-1e-5:best=v;bs=step;wait=0;state=copy.deepcopy(net.state_dict())
                    else:wait+=1
                    if wait>=6:break
            if state is not None:net.load_state_dict(state)
            m.nets.append(net);m.best_steps.append(bs if xv is not None else steps)
    m.seconds=time.perf_counter()-t0;return m

def search():
    OUT.mkdir(parents=True,exist_ok=True);d,dates,X,S,mt=dataset();delta=(mt.y-mt['last']).to_numpy();rng=np.random.default_rng(42);fids=rng.choice(d.FID,512,replace=False);rows=[];preds=[];splits=[]
    (OUT/'experiment_design.json').write_text(json.dumps({'stage':'D, exploratory multi-lead and forecast-temperature features','validation':VDATES,'candidates':CONFIGS,
      'training_leads_days':[24,180],'search_training_FID':512,'final_training_row_cap':240000,'weather':'past complete years climatology + past30-day anomaly decayed with 30-day timescale; no actual future weather',
      'optimizer':'fixed update counts avoid epoch-count changes when training size grows; select steps only on pre-2021 validation',
      'endpoints':[30,60,90],'primary_endpoint':90,'2021_already_seen':True},indent=2),encoding='utf-8')
    for target in VDATES:
        va=((mt.role=='validation')&(mt.target==target)).to_numpy();origin=mt.loc[va,'origin'].iloc[0]
        tr=((mt.role=='train')&(mt.target<=origin)&mt.FID.isin(fids)).to_numpy();assert mt.loc[tr,'target'].max()<=origin
        splits.append({'target':target,'origin':origin,'training_label_max':mt.loc[tr,'target'].max(),'training_rows':int(tr.sum())})
        for c in CONFIGS:
            m=fit(c,X.loc[tr],delta[tr],S[tr],X.loc[va],delta[va],S[va]);p=mt.loc[va,'last'].to_numpy()+m.predict(X.loc[va],S[va])
            rows.append({'id':c['id'],'model':c['name'],'target':target,**metrics(mt.loc[va,'y'],p),'steps':m.best_steps[0] if m.best_steps else None,'seconds':m.seconds})
            a=mt.loc[va,['FID','target','y']].copy();a['id']=c['id'];a['pred']=p;preds.append(a)
            print('MULTILEAD_VALIDATION',target,c['id'],round(rows[-1]['RMSE'],3),m.best_steps,flush=True)
        save(pd.DataFrame(rows),'validation.csv')
    pv=pd.concat(preds,ignore_index=True);save(pv,'validation_predictions.csv.gz');save(pd.DataFrame(splits),'validation_splits.csv');scores=[]
    for c in CONFIGS:
        a=pv[pv.id==c['id']];scores.append({'id':c['id'],'model':c['name'],**metrics(a.y,a.pred)})
    sc=pd.DataFrame(scores);save(sc,'validation_summary.csv');selected=[]
    for name,a in sc.groupby('model',sort=False):
        pick=a.sort_values('RMSE').iloc[0].id;c=copy.deepcopy(next(c for c in CONFIGS if c['id']==pick));es=pd.DataFrame(rows).loc[lambda x:x.id==pick,'steps'].dropna();c['steps']=int(np.median(es)) if len(es) else None;selected.append(c)
    lock={'selected':selected,'features':list(X.columns),'validation_dates':VDATES,'frozen_training_label_cutoff':'2020-11-01','final_row_cap':240000,
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/multihorizon.py',ROOT/'src/improvement.py',ROOT/'src/models.py',ROOT/'src/core.py',ROOT/'data/points_base.csv',ROOT/'data/climate_daily.csv',ROOT/'data/climate_metadata.json']}}
    (OUT/'locked_selection.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')

def final():
    raw=(OUT/'locked_selection.json').read_bytes();lock=json.loads(raw)
    for p,h in lock['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    d,dates,X,S,mt=dataset();delta=(mt.y-mt['last']).to_numpy();ix=np.flatnonzero(mt.role=='train');rng=np.random.default_rng(42)
    if len(ix)>lock['final_row_cap']:ix=np.sort(rng.choice(ix,lock['final_row_cap'],replace=False))
    assert mt.loc[ix,'target'].max()<='2020-11-01';out=[];scores=[]
    test=(mt.role=='test').to_numpy()
    for c in lock['selected']:
        m=fit(c,X.loc[ix],delta[ix],S[ix],steps=c['steps'],members=5);p=mt.loc[test,'last'].to_numpy()+m.predict(X.loc[test],S[test]);a=mt.loc[test].copy();a['model']=c['name'];a['pred']=p;out.append(a)
        for horizon,b in a.groupby('protocol_horizon'):
            scores.append({'model':c['name'],'horizon_min_days':horizon,'actual_lead_min':b.horizon_days.min(),'actual_lead_max':b.horizon_days.max(),'steps':c['steps'],'train_seconds':m.seconds,**metrics(b.y,b.pred)})
            print('MULTILEAD_FINAL',scores[-1],flush=True)
        if c['kind']=='neural':
            with (OUT/(c['name']+'.pkl')).open('wb') as f:pickle.dump(m,f,protocol=5)
        save(pd.DataFrame(scores),'metrics.csv')
    save(pd.concat(out,ignore_index=True),'predictions.csv.gz');(OUT/'selection_sha256.txt').write_text(hashlib.sha256(raw).hexdigest()+'\n')

if __name__=='__main__':
    import sys
    sys.modules['multihorizon']=sys.modules[__name__];Predictor.__module__='multihorizon'
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['search','final']);a=p.parse_args();search() if a.stage=='search' else final()
