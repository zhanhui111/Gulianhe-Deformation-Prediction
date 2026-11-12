"""Pre-2021 model selection for the 90-day panel; never filter by test errors.

Run with run_improvement.py. Original verified results are retained as baseline.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from core import ROOT, load_points, temporal_panel, metric_xy, metrics
from models import Scale, vector, put_vector

OUT=ROOT/'results/improvement'
VALIDATION_TARGETS=['2020-06-10','2020-08-09','2020-11-01']
COMPACT=['harmonic_change','trend_change','annual_change','annual_level_change',
         'local_annual_change','target_sin','target_cos','horizon_days']
CONFIGS=[
 {'name':'SeasonalRidge','kind':'ridge','alpha':10},
 {'name':'SeasonalRidge','kind':'ridge','alpha':1000},
 {'name':'XGBoost','kind':'xgb','depth':2,'leaf':10},
 {'name':'XGBoost','kind':'xgb','depth':4,'leaf':10},
 {'name':'RandomForest','kind':'rf','leaf':10},
 {'name':'RandomForest','kind':'rf','leaf':30},
 {'name':'Calendar-GA-PSO-BP','kind':'neural','hidden':8,'residual':False,'ga':True},
 {'name':'Calendar-GA-PSO-BP','kind':'neural','hidden':24,'residual':False,'ga':True},
 {'name':'Seasonal-GA-PSO-BP','kind':'neural','hidden':8,'residual':True,'ga':True},
 {'name':'Seasonal-GA-PSO-BP','kind':'neural','hidden':24,'residual':True,'ga':True},
 {'name':'Seasonal-BP','kind':'neural','hidden':8,'residual':True,'ga':False},
 {'name':'Seasonal-BP','kind':'neural','hidden':24,'residual':True,'ga':False},
 {'name':'LSTM','kind':'neural','hidden':8,'residual':True,'ga':False,'sequence':True},
 {'name':'LSTM','kind':'neural','hidden':24,'residual':True,'ga':False,'sequence':True},
 {'name':'BP','kind':'neural','hidden':8,'residual':False,'ga':False},
 {'name':'BP','kind':'neural','hidden':24,'residual':False,'ga':False},
 {'name':'LSTM','kind':'neural','hidden':8,'residual':False,'ga':False,'sequence':True},
 {'name':'LSTM','kind':'neural','hidden':24,'residual':False,'ga':False,'sequence':True},
 {'name':'SeasonalNaive','kind':'naive'},
]
for i,c in enumerate(CONFIGS):
    c['id']=f'C{i:02d}'
    if c['name']=='Seasonal-BP':c['name']='BP'

def panel(data_override=None):
    d,dates=load_points() if data_override is None else data_override
    X,S,mt=temporal_panel(d,dates,90)
    xy=metric_xy(d);ix=NearestNeighbors(n_neighbors=17).fit(xy).kneighbors(xy,return_distance=False)[:,1:]
    values=d[dates].to_numpy();times=pd.to_datetime([c[2:] for c in dates]);frames=[]
    for (origin,target),rows in mt.groupby(['origin','target'],sort=False):
        z=X.loc[rows.index].copy();od=pd.Timestamp(origin);td=pd.Timestamp(target)
        keep=np.asarray(times<=od);tt=(times[keep]-times[0]).days.to_numpy()/365.25;Y=values[:,keep]
        A=np.column_stack([np.ones(len(tt)),tt,np.sin(2*np.pi*tt),np.cos(2*np.pi*tt)])
        co=np.linalg.solve(A.T@A+np.diag([0,0,1,1]),A.T@Y.T)
        tf=(td-times[0]).days/365.25
        season=(np.array([np.sin(2*np.pi*tf),np.cos(2*np.pi*tf)])-A[-1,2:])@co[2:]
        # Interpolation is confined to dates already observed at the origin.
        histdays=(times[keep]-times[0]).days.to_numpy()
        py=(td-pd.Timedelta(days=365)-times[0]).days
        po=(od-pd.Timedelta(days=365)-times[0]).days
        prior=np.array([np.interp(py,histdays,v) for v in Y])
        prior_o=np.array([np.interp(po,histdays,v) for v in Y])
        available=(py>=histdays[0] and po>=histdays[0] and py<=histdays[-1])
        annual=prior-prior_o if available else season
        z['target_sin']=np.sin(2*np.pi*td.dayofyear/365.25)
        z['target_cos']=np.cos(2*np.pi*td.dayofyear/365.25)
        z['harmonic_change']=season
        z['trend_change']=co[1]*(td-od).days/365.25
        z['annual_change']=annual
        z['annual_level_change']=prior-Y[:,-1] if py>=histdays[0] else season
        z['annual_available']=float(available)
        z['history_fit_RMSE']=np.sqrt(np.mean((Y-(A@co).T)**2,axis=1))
        z['local_last']=Y[:,-1][ix].mean(1)
        z['local_trend']=co[1][ix].mean(1)
        z['local_annual_change']=annual[ix].mean(1)
        z['easting_km']=xy[:,0]/1000;z['northing_km']=xy[:,1]/1000
        frames.append(z)
    X=pd.concat(frames).sort_index()
    assert len(X)==len(mt) and np.isfinite(X.to_numpy()).all()
    spl=pd.read_csv(ROOT/'results/tables/splits.csv').set_index('FID')
    mt['block']=mt.FID.map(spl.block)
    return X,S,mt

def baseline(X,y,alpha=100):
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha));m.fit(X[COMPACT],y);return m

class Network(nn.Module):
    def __init__(self,d,h,sequence=False):
        super().__init__();self.sequence=sequence
        if sequence:self.lstm=nn.LSTM(2,32,batch_first=True)
        self.hidden=nn.Linear(d+(32 if sequence else 0),h);self.out=nn.Linear(h,1)
    def forward(self,x,s=None):
        if self.sequence:
            a,_=self.lstm(s);x=torch.cat([x,a[:,-1]],dim=1)
        return self.out(torch.tanh(self.hidden(x))).squeeze(-1)

def initialise(net,X,y,seed):
    """GA then PSO, arbitrary hidden width, both biases included."""
    rng=np.random.default_rng(seed);h=net.hidden.out_features;d=X.shape[1]
    j=rng.choice(len(y),min(len(y),2048),replace=False);X=X[j];y=y[j]
    L=d*h+2*h+1;P=24;pop=rng.uniform(-.5,.5,(P,L)).astype('float32')
    def fitness(p):
        W=p[:,:d*h].reshape(-1,h,d);b=p[:,d*h:d*h+h];v=p[:,d*h+h:d*h+2*h]
        a=np.tanh(np.einsum('phd,nd->pnh',W,X,optimize=True)+b[:,None,:])
        pred=np.einsum('pnh,ph->pn',a,v)+p[:,-1,None]
        return ((pred-y)**2).mean(1)+.0001*(p*p).mean(1)
    f=fitness(pop)
    for _ in range(30):
        order=np.argsort(f);new=[pop[k].copy() for k in order[:2]]
        def choose():
            a,b=rng.choice(P,2,replace=False);return pop[a if f[a]<f[b] else b]
        while len(new)<P:
            a,b=choose(),choose();c=a.copy()
            if rng.random()<.8:
                w=rng.random(L);c=w*a+(1-w)*b
            m=rng.random(L)<.08;c[m]+=rng.normal(0,.3,m.sum());new.append(np.clip(c,-3,3))
        pop=np.asarray(new,dtype='float32');f=fitness(pop)
    vel=np.zeros_like(pop);best=pop.copy();bf=f.copy();gb=best[bf.argmin()].copy()
    for t in range(30):
        w=.9-(.9-.4)*t/29
        vel=np.clip(w*vel+1.5*rng.random(pop.shape)*(best-pop)+1.5*rng.random(pop.shape)*(gb-pop),-1,1)
        pop=np.clip(pop+.5*vel,-3,3).astype('float32');f=fitness(pop)
        take=f<bf;best[take]=pop[take];bf[take]=f[take];gb=best[bf.argmin()].copy()
    put_vector(net,gb);assert np.array_equal(vector(net),gb)

class Model:
    def predict_members(self,X,S):
        if self.kind=='naive':return np.array([X.annual_change.to_numpy()])
        if self.kind=='ridge':return np.array([self.est.predict(X[COMPACT])])
        if self.kind!='neural':return np.array([self.est.predict(X)])
        base=self.base.predict(X[COMPACT]) if self.base is not None else np.zeros(len(X))
        xx=torch.tensor(self.scale.x(X.to_numpy()));ss=torch.tensor(self.scale.seq(S)) if self.sequence else None
        out=[]
        for net in self.nets:
            net.eval();pred=[]
            with torch.no_grad():
                for j in range(0,len(X),4096):pred.append(net(xx[j:j+4096],ss[j:j+4096] if ss is not None else None).numpy())
            out.append(base+np.concatenate(pred)*self.scale.ys+self.scale.ym)
        return np.asarray(out)
    def predict(self,X,S):return self.predict_members(X,S).mean(0)

def fit(c,X,y,S,Xv=None,yv=None,Sv=None,epochs=None,members=1):
    start=time.perf_counter();m=Model();m.kind=c['kind'];m.epochs=[];m.sequence=c.get('sequence',False)
    if m.kind=='naive':pass
    elif m.kind=='ridge':m.est=baseline(X,y,c['alpha'])
    elif m.kind=='rf':m.est=RandomForestRegressor(n_estimators=200,min_samples_leaf=c['leaf'],max_features=.8,n_jobs=2,random_state=42).fit(X,y)
    elif m.kind=='xgb':m.est=XGBRegressor(n_estimators=350,max_depth=c['depth'],min_child_weight=c['leaf'],learning_rate=.04,reg_lambda=10,subsample=.8,colsample_bytree=.8,n_jobs=2,random_state=42).fit(X,y)
    else:
        m.base=baseline(X,y,100) if c.get('residual') else None
        r=y-(m.base.predict(X[COMPACT]) if m.base is not None else 0)
        rv=yv-(m.base.predict(Xv[COMPACT]) if m.base is not None else 0) if Xv is not None else None
        m.scale=Scale().fit(X.to_numpy(),r,S);xx=torch.tensor(m.scale.x(X.to_numpy()));yy=torch.tensor(m.scale.y(r))
        ss=torch.tensor(m.scale.seq(S)) if m.sequence else None
        xv=torch.tensor(m.scale.x(Xv.to_numpy())) if Xv is not None else None
        yval=torch.tensor(m.scale.y(rv)) if Xv is not None else None
        sv=torch.tensor(m.scale.seq(Sv)) if Xv is not None and m.sequence else None
        m.nets=[]
        for se in range(42,42+members):
            torch.manual_seed(se);net=Network(X.shape[1],c['hidden'],m.sequence)
            if c.get('ga'):initialise(net,xx.numpy(),yy.numpy(),se)
            opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001)
            gen=torch.Generator().manual_seed(se);best=np.inf;bestep=1;state=None;wait=0
            for ep in range(1,(epochs or 100)+1):
                net.train()
                for j in torch.randperm(len(yy),generator=gen).split(1024):
                    opt.zero_grad();loss=((net(xx[j],ss[j] if ss is not None else None)-yy[j])**2).mean()
                    loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                if xv is not None:
                    net.eval()
                    with torch.no_grad(): vl=float(((net(xv,sv)-yval)**2).mean())
                    if vl<best-1e-5:best=vl;bestep=ep;state=copy.deepcopy(net.state_dict());wait=0
                    else:wait+=1
                    if wait>=15:break
            if state is not None:net.load_state_dict(state)
            m.epochs.append(bestep if xv is not None else epochs);m.nets.append(net)
    m.seconds=time.perf_counter()-start
    return m

def save(df,name):df.to_csv(OUT/name,index=False,float_format='%.10g')

def search():
    OUT.mkdir(parents=True,exist_ok=True);X,S,mt=panel();delta=(mt.y-mt['last']).to_numpy()
    (OUT/'experiment_design.json').write_text(json.dumps({'primary_selection':'minimum pre-2021 validation RMSE across Calendar/Seasonal GA-PSO-BP candidates',
      'primary_endpoint':'all 12055 points, pooled 11 targets in 2021, actual leads 96-132 days',
      'candidates':CONFIGS,'validation_targets':VALIDATION_TARGETS,'features':list(X.columns),
      'search_seed':42,'quality_filter':'pre-origin harmonic-fit RMSE; thresholds from pre-2021 training only',
      'development_amendment':'Before new 2021 evaluation: initial 14 candidates showed poor seasonal-branch validation. Add direct BP/LSTM candidates and SeasonalNaive; choose GA architecture by past validation. Full search reproducible from this source.',
      'stop_rule':'no revisions after first new 2021 evaluation; all selection uses pre-2021 validation'},indent=2),encoding='utf-8')
    # Fixed FID sample reduces only the search training cost; full validation retained.
    fid=np.sort(mt.FID.unique());chosen=np.random.default_rng(42).choice(fid,min(2048,len(fid)),replace=False)
    rows=[];predictions=[];splitrows=[]
    if (OUT/'validation.csv').exists() and (OUT/'validation_predictions.csv').exists():
        old=pd.read_csv(OUT/'validation.csv');old['name']=old.id.map({c['id']:c['name'] for c in CONFIGS})
        rows=old.to_dict('records');predictions=[pd.read_csv(OUT/'validation_predictions.csv')]
    for target in VALIDATION_TARGETS:
        va=(mt.target==target).to_numpy();origin=mt.loc[va,'origin'].iloc[0]
        tr=((mt.target<=origin)&mt.FID.isin(chosen)).to_numpy()
        assert mt.loc[tr,'target'].max()<=origin and va.any() and tr.any()
        splitrows.append({'target':target,'origin':origin,'max_training_target':mt.loc[tr,'target'].max(),'train_rows':int(tr.sum()),'val_rows':int(va.sum())})
        for c in CONFIGS:
            if any(r['id']==c['id'] and r['target']==target for r in rows):continue
            m=fit(c,X.loc[tr],delta[tr],S[tr],X.loc[va],delta[va],S[va])
            p=mt.loc[va,'last'].to_numpy()+m.predict(X.loc[va],S[va])
            rows.append({'id':c['id'],'name':c['name'],'target':target,'origin':origin,**metrics(mt.loc[va,'y'],p),'epoch':m.epochs[0] if m.epochs else None,'seconds':m.seconds})
            print('VALIDATION',target,c['id'],c['name'],round(rows[-1]['RMSE'],4),m.epochs,flush=True)
            f=mt.loc[va,['FID','block','target','y']].copy();f['id']=c['id'];f['pred']=p;predictions.append(f)
        save(pd.DataFrame(rows),'validation.csv')
    save(pd.DataFrame(splitrows),'validation_splits.csv')
    pv=pd.concat(predictions,ignore_index=True);save(pv,'validation_predictions.csv')
    scores=[]
    for c in CONFIGS:
        a=pv[pv.id==c['id']];scores.append({'id':c['id'],'name':c['name'],**metrics(a.y,a.pred)})
    sc=pd.DataFrame(scores);save(sc,'validation_summary.csv');selected=[]
    for name,a in sc.groupby('name',sort=False):
        pick=a.sort_values('RMSE').iloc[0]['id'];c=copy.deepcopy(next(c for c in CONFIGS if c['id']==pick))
        es=pd.DataFrame(rows).loc[lambda z:z.id==pick,'epoch'].dropna()
        c['epochs']=max(1,int(np.median(es))) if len(es) else None;selected.append(c)
    primary_name=sc[sc.name.isin(['Calendar-GA-PSO-BP','Seasonal-GA-PSO-BP'])].sort_values('RMSE').iloc[0]['name']
    matched=copy.deepcopy(next(c for c in selected if c['name']==primary_name))
    matched.update(name='Matched BP',id='AB_BP',ga=False);selected.append(matched)
    train=(mt.target<='2020-11-01').to_numpy()
    qc={str(q):float(np.quantile(X.loc[train,'history_fit_RMSE'],q)) for q in [.5,.75,.9]}
    lock={'protocol':'temporal_90d','validation_targets':VALIDATION_TARGETS,'search_training_FID_count':len(chosen),
      'selection':'pooled validation RMSE, targets strictly after each training cutoff; no 2021 labels used for fitting or selection',
      'candidate_configs':CONFIGS,'selected':selected,'quality_thresholds_history_RMSE_mm':qc,
      'primary_model':primary_name,
      'feature_names':list(X.columns),'final_train_target_max':'2020-11-01','final_seeds':[42,43,44,45,46],
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [Path(__file__),ROOT/'src/core.py',ROOT/'src/models.py',ROOT/'data/points_base.csv',ROOT/'data/climate_daily.csv',ROOT/'data/climate_metadata.json']},
      'note':'2021 baseline results were already examined before designing this exploratory extension. External confirmation remains needed.'}
    (OUT/'locked_selection.json').write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding='utf-8')
    print('LOCKED',hashlib.sha256((OUT/'locked_selection.json').read_bytes()).hexdigest(),flush=True)

def final():
    raw=(OUT/'locked_selection.json').read_bytes();lock=json.loads(raw)
    for path,digest in lock['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    X,S,mt=panel();y=mt.y.to_numpy();last=mt['last'].to_numpy();delta=y-last
    tr=(mt.target<='2020-11-01').to_numpy();te=((mt.target>='2021-01-01')&(mt.origin>='2020-11-01')).to_numpy()
    assert mt.loc[tr,'target'].max()<=mt.loc[te,'origin'].min()
    preds=[];res=[]
    for c in lock['selected']:
        m=fit(c,X.loc[tr],delta[tr],S[tr],epochs=c['epochs'],members=5)
        members=m.predict_members(X.loc[te],S[te])+last[te];p=members.mean(0)
        a=mt.loc[te].copy();a['model']=c['name'];a['pred']=p;a['ensemble_sd']=members.std(0,ddof=1) if len(members)>1 else np.nan
        a['history_fit_RMSE']=X.loc[te,'history_fit_RMSE'].to_numpy();preds.append(a)
        res.append({'model':c['name'],'config_id':c['id'],'selected_epochs':c['epochs'],'members':len(members),'seconds':m.seconds,'parameters_per_member':sum(p.numel() for p in m.nets[0].parameters()) if c['kind']=='neural' else None,**metrics(y[te],p)})
        if c['kind']=='neural' or c['kind']=='ridge':
            with (OUT/(c['name']+'.pkl')).open('wb') as f:pickle.dump(m,f,protocol=5)
        save(pd.DataFrame(res),'metrics.csv');save(pd.concat(preds,ignore_index=True),'predictions.csv')
        print('FINAL',c['name'],metrics(y[te],p),'seconds',round(m.seconds,1),flush=True)
    assert hashlib.sha256(raw).hexdigest()==hashlib.sha256((OUT/'locked_selection.json').read_bytes()).hexdigest()
    (OUT/'selection_sha256.txt').write_text(hashlib.sha256(raw).hexdigest()+'\n',encoding='utf-8')

if __name__=='__main__':
    import sys
    sys.modules['improvement']=sys.modules[__name__]
    Network.__module__='improvement';Model.__module__='improvement'
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['search','final']);a=p.parse_args()
    search() if a.stage=='search' else final()
