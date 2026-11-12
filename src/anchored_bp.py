"""Phase G: shared-network future-minus-present increments, only own method.

Competitors remain fixed. Plain BP is an exact architectural ablation, not tuned.
The common output bias cancels in a difference, so it is deliberately omitted.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,pickle,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from core import ROOT,metrics,load_points
from models import Scale,put_vector
from multihorizon import dataset,make_pair,Thermal

OUT=ROOT/'results/anchored_bp';BASE=ROOT/'results/multihorizon'
TARGETS=['2020-01-06','2020-02-11','2020-03-06','2020-04-11','2020-05-05','2020-06-10','2020-07-04','2020-08-09','2020-09-02','2020-10-08','2020-11-01']

def save(df,name):df.to_csv(OUT/name,index=False,float_format='%.10g')
def zero_inputs(X):
    z=X.copy();z['horizon_days']=0
    for c in ['expected_thaw_DD','expected_freeze_DD','expected_sqrt_thaw180_change','expected_sqrt_freeze180_change']:z[c]=0
    z['expected_temp_target30']=z.temp_30
    z['target_sin_1']=z.origin_sin;z['target_cos_1']=z.origin_cos
    z['target_sin_2']=2*z.origin_sin*z.origin_cos;z['target_cos_2']=z.origin_cos**2-z.origin_sin**2
    return z
class Net(nn.Module):
    def __init__(self,d,h):super().__init__();self.hidden=nn.Linear(d,h);self.out=nn.Linear(h,1,bias=False)
    def forward(self,x,z):return self.out(torch.tanh(self.hidden(x))-torch.tanh(self.hidden(z))).squeeze(-1)
class Predictor:
    def predict(self,X):
        xx=torch.tensor(self.scale.x(np.asarray(X)));zz=torch.tensor(self.scale.x(zero_inputs(X).to_numpy()));out=[]
        for net in self.nets:
            net.eval();p=[]
            with torch.no_grad():
                for j in range(0,len(X),4096):p.append(net(xx[j:j+4096],zz[j:j+4096]).numpy())
            out.append(np.concatenate(p)*self.scale.ys)
        return np.mean(out,axis=0)

def init(net,X,Z,y,seed):
    rng=np.random.default_rng(seed);j=rng.choice(len(y),min(len(y),2048),replace=False);X=X[j];Z=Z[j];y=y[j]
    d=X.shape[1];h=net.hidden.out_features;L=d*h+2*h;P=24;pop=rng.uniform(-.5,.5,(P,L)).astype('float32')
    def fitness(p):
        W=p[:,:d*h].reshape(-1,h,d);b=p[:,d*h:d*h+h];v=p[:,d*h+h:]
        a=np.tanh(np.einsum('phd,nd->pnh',W,X,optimize=True)+b[:,None,:])-np.tanh(np.einsum('phd,nd->pnh',W,Z,optimize=True)+b[:,None,:])
        pred=np.einsum('pnh,ph->pn',a,v);return ((pred-y)**2).mean(1)+.0001*(p*p).mean(1)
    f=fitness(pop)
    for _ in range(30):
        order=np.argsort(f);new=[pop[i].copy() for i in order[:2]]
        def choose():
            a,b=rng.choice(P,2,replace=False);return pop[a if f[a]<f[b] else b]
        while len(new)<P:
            a,b=choose(),choose();c=a.copy()
            if rng.random()<.8:w=rng.random(L);c=w*a+(1-w)*b
            mask=rng.random(L)<.08;c[mask]+=rng.normal(0,.3,mask.sum());new.append(np.clip(c,-3,3))
        pop=np.asarray(new,dtype='float32');f=fitness(pop)
    vel=np.zeros_like(pop);best=pop.copy();bf=f.copy();gb=best[bf.argmin()].copy()
    for t in range(30):
        inertia=.9-.5*t/29;vel=np.clip(inertia*vel+1.5*rng.random(pop.shape)*(best-pop)+1.5*rng.random(pop.shape)*(gb-pop),-1,1)
        pop=np.clip(pop+.5*vel,-3,3).astype('float32');f=fitness(pop);take=f<bf;best[take]=pop[take];bf[take]=f[take];gb=best[bf.argmin()].copy()
    put_vector(net,gb)

def fit(X,y,hidden=5,steps=600,members=1,ga=True,Xv=None,yv=None):
    m=Predictor();Z=zero_inputs(X);m.scale=Scale().fit(np.vstack([np.asarray(X),np.asarray(Z)]),np.r_[y,y]);m.scale.ym=0
    xx=torch.tensor(m.scale.x(np.asarray(X)));zz=torch.tensor(m.scale.x(np.asarray(Z)));yy=torch.tensor((y/m.scale.ys).astype('float32'))
    xv=torch.tensor(m.scale.x(np.asarray(Xv))) if Xv is not None else None;zv=torch.tensor(m.scale.x(zero_inputs(Xv).to_numpy())) if Xv is not None else None
    yvt=torch.tensor((yv/m.scale.ys).astype('float32')) if Xv is not None else None;m.nets=[];m.steps=[]
    for seed in range(42,42+members):
        torch.manual_seed(seed);net=Net(X.shape[1],hidden)
        if ga:init(net,xx.numpy(),zz.numpy(),yy.numpy(),seed)
        opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001);gen=torch.Generator().manual_seed(seed);best=np.inf;bs=steps;wait=0;state=None
        for step in range(1,steps+1):
            net.train();ix=torch.randint(len(yy),(512,),generator=gen);opt.zero_grad();loss=((net(xx[ix],zz[ix])-yy[ix])**2).mean();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            if Xv is not None and step%20==0:
                net.eval()
                with torch.no_grad():v=float(((net(xv,zv)-yvt)**2).mean())
                if v<best-1e-5:best=v;bs=step;wait=0;state=copy.deepcopy(net.state_dict())
                else:wait+=1
                if wait>=6:break
        if state is not None:net.load_state_dict(state)
        m.nets.append(net);m.steps.append(bs)
    return m

def search():
    OUT.mkdir(parents=True,exist_ok=True);d,dates,X,S,mt=dataset();thermal=Thermal(d);cases={}
    for target in TARGETS:
        for horizon in [30,60,90]:
            eligible=[c for c in dates if pd.Timestamp(c[2:])<=pd.Timestamp(target)-pd.Timedelta(days=horizon)]
            if len(eligible)>=4:
                origin=pd.Timestamp(eligible[-1][2:]).strftime('%Y-%m-%d');cases[(origin,target)]=make_pair(d,dates,origin,target,thermal)
    fids=np.random.default_rng(42).choice(d.FID,512,replace=False);delta=(mt.y-mt['last']).to_numpy();rows=[];preds=[]
    (OUT/'experiment_design.json').write_text(json.dumps({'stage':'G, only own GA-PSO-BP modified after user scope clarification','competitors':'fixed phase-D models; no further tuning or calibration',
      'architecture':'one shared tanh network F(state,future)-F(state,present), common output bias omitted because it cancels; target scale has zero offset',
      'validation_targets':TARGETS,'hidden_candidates':[5,16],'max_updates':600,'seed_search':42,'seeds_final':[42,43,44,45,46],
      'secondary_shrink':'nonnegative scalar from mature-history weighted past OOF, no 2021 residuals','matched_ablation':'same width, steps, scaling, data and shrink; only GA-PSO initialization removed','2021_already_seen':True},indent=2),encoding='utf-8')
    for origin in sorted(set(o for o,t in cases)):
        tr=((mt.role=='train')&(mt.target<=origin)&mt.FID.isin(fids)).to_numpy();assert mt.loc[tr,'target'].max()<=origin
        keys=[k for k in cases if k[0]==origin];vx=pd.concat([cases[k][0] for k in keys],ignore_index=True);vm=pd.concat([cases[k][2] for k in keys],ignore_index=True)
        for hidden in [5,16]:
            m=fit(X.loc[tr],delta[tr],hidden=hidden,Xv=vx,yv=(vm.y-vm['last']).to_numpy());p=vm['last'].to_numpy()+m.predict(vx)
            rows.append({'origin':origin,'hidden':hidden,'steps':m.steps[0],**metrics(vm.y,p)})
            a=vm.copy();a['hidden']=hidden;a['pred']=p;preds.append(a)
        print('ANCHORED_VALIDATION',origin,'cases',len(keys),rows[-2]['RMSE'],rows[-1]['RMSE'],flush=True)
    pv=pd.concat(preds,ignore_index=True);save(pd.DataFrame(rows),'validation.csv');save(pv,'validation_predictions.csv.gz');summary=[]
    for hidden,a in pv.groupby('hidden'):summary.append({'hidden':hidden,**metrics(a.y,a.pred)})
    sc=pd.DataFrame(summary);save(sc,'validation_summary.csv');hidden=int(sc.sort_values('RMSE').iloc[0].hidden);steps=int(np.median([r['steps'] for r in rows if r['hidden']==hidden]))
    # Match the final five-member estimator for calibration; no competitor fits.
    calibration=[]
    for origin in sorted(set(o for o,t in cases)):
        tr=((mt.role=='train')&(mt.target<=origin)&mt.FID.isin(fids)).to_numpy();keys=[k for k in cases if k[0]==origin]
        vx=pd.concat([cases[k][0] for k in keys],ignore_index=True);vm=pd.concat([cases[k][2] for k in keys],ignore_index=True)
        m=fit(X.loc[tr],delta[tr],hidden=hidden,steps=steps,members=5);vm['pred']=vm['last'].to_numpy()+m.predict(vx)
        first=mt.loc[tr,'target'].min();span=(pd.Timestamp(origin)-pd.Timestamp(first)).days;vm['maturity_weight']=min(span/365,1)**4;calibration.append(vm)
    cv=pd.concat(calibration,ignore_index=True);save(cv,'calibration_predictions.csv.gz');q=(cv.pred-cv['last']).to_numpy();r=(cv.y-cv['last']).to_numpy();w=cv.maturity_weight.to_numpy()
    alpha=float(np.clip(np.sum(w*q*r)/max(np.sum(w*q*q),1e-12),0,1))
    lock={'hidden':hidden,'steps':steps,'shrink_alpha':alpha,'maturity_weight':'min(training_label_span_days/365,1)^4; weights downweight immature historical fits compared with the >1-year final training set',
      'primary':'Shrink Anchored GA-PSO-BP','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/anchored_bp.py',ROOT/'src/multihorizon.py',ROOT/'src/improvement.py',ROOT/'src/core.py',ROOT/'src/models.py',ROOT/'data/points_base.csv',ROOT/'data/climate_daily.csv',ROOT/'data/climate_metadata.json']}}
    (OUT/'locked_selection.json').write_text(json.dumps(lock,indent=2),encoding='utf-8');print('ANCHORED_LOCKED',hidden,steps,alpha,flush=True)

def final():
    raw=(OUT/'locked_selection.json').read_bytes();lock=json.loads(raw)
    for p,h in lock['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    d,dates,X,S,mt=dataset();tr=np.flatnonzero(mt.role=='train');rng=np.random.default_rng(42)
    if len(tr)>240000:tr=np.sort(rng.choice(tr,240000,replace=False))
    test=(mt.role=='test').to_numpy();output=[];rows=[]
    for ga,label in [(True,'Anchored GA-PSO-BP'),(False,'Matched anchored BP')]:
        m=fit(X.loc[tr],(mt.y-mt['last']).to_numpy()[tr],hidden=lock['hidden'],steps=lock['steps'],members=5,ga=ga);delta=m.predict(X.loc[test]);a=mt.loc[test].copy()
        with (OUT/(label+'.pkl')).open('wb') as f:pickle.dump(m,f,protocol=5)
        for shrink,name in [(1,label),(lock['shrink_alpha'],'Shrink '+label)]:
            b=a.copy();b['model']=name;b['pred']=b['last'].to_numpy()+shrink*delta;output.append(b)
            for horizon,c in b.groupby('protocol_horizon'):
                row={'model':name,'horizon_min_days':horizon,**metrics(c.y,c.pred)};rows.append(row);print('ANCHORED_FINAL',row,flush=True)
    save(pd.concat(output,ignore_index=True),'predictions.csv.gz');save(pd.DataFrame(rows),'metrics.csv');(OUT/'selection_sha256.txt').write_text(hashlib.sha256(raw).hexdigest()+'\n')

if __name__=='__main__':
    import sys
    sys.modules['anchored_bp']=sys.modules[__name__];Net.__module__='anchored_bp';Predictor.__module__='anchored_bp'
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['search','final']);a=p.parse_args();search() if a.stage=='search' else final()
