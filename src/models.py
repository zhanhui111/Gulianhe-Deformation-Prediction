"""Small neural baselines with train-only preprocessing and correct Adam updates.

GA and PSO initialise a five-hidden-unit BP network; this is an explicit new
Python implementation, not a bit-for-bit reproduction of MATLAB newff/train.
"""
from __future__ import annotations
import copy,time
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
torch.set_num_threads(2)
SETTINGS={'hidden':5,'population':24,'ga_generations':30,'pso_generations':30,
 'crossover_probability':.8,'mutation_probability':.08,'c1':1.5,'c2':1.5,'w_start':.9,'w_end':.4,
 'velocity_limit':1.,'position_limit':3.,'fitness_sample_max':2048,'epochs_max':240,
 'batch_size':512,'learning_rate':.003,'weight_decay':.0001,'patience':25,'ensemble_seeds':5}

class Scale:
    def fit(self,X,y,S=None):
        self.med=np.nanmedian(X,axis=0);self.med=np.nan_to_num(self.med)
        z=np.where(np.isfinite(X),X,self.med);self.mu=z.mean(0);self.sd=z.std(0);self.sd[self.sd<1e-8]=1
        self.ym=float(y.mean());self.ys=max(float(y.std()),1e-6)
        self.seq_sd=max(float(S[:,:,0].std()),1e-6) if S is not None else 1
        return self
    def x(self,X):return ((np.where(np.isfinite(X),X,self.med)-self.mu)/self.sd).astype('float32')
    def y(self,y):return ((y-self.ym)/self.ys).astype('float32')
    def seq(self,S):
        s=S.astype('float32').copy();s[:,:,0]/=self.seq_sd;return s

class Net(nn.Module):
    def __init__(self,d,sequence=False):
        super().__init__();self.sequence=sequence
        if sequence:self.lstm=nn.LSTM(2,16,batch_first=True)
        self.hidden=nn.Linear(d+(16 if sequence else 0),5);self.out=nn.Linear(5,1)
    def forward(self,x,s=None):
        if self.sequence:
            z,_=self.lstm(s);x=torch.cat([x,z[:,-1]],dim=1)
        return self.out(torch.tanh(self.hidden(x))).squeeze(-1)

def vector(net):return np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])

def put_vector(net,v):
    pos=0
    with torch.no_grad():
        for p in net.parameters():
            n=p.numel();p.copy_(torch.tensor(v[pos:pos+n].reshape(tuple(p.shape)),dtype=p.dtype));pos+=n
    assert pos==len(v)

def meta_initialise(net,X,y,seed,mode='GA-PSO-BP'):
    rng=np.random.default_rng(seed);d=X.shape[1];h=5
    j=rng.choice(len(y),min(len(y),SETTINGS['fitness_sample_max']),replace=False);X=X[j];y=y[j]
    L=d*h+h+h+1;P=SETTINGS['population'];pop=rng.uniform(-.5,.5,(P,L)).astype('float32')
    def fitness(p):
        W=p[:,:d*h].reshape(-1,h,d);b=p[:,d*h:d*h+h];v=p[:,d*h+h:d*h+2*h];c=p[:,-1]
        z=np.tanh(np.einsum('phd,nd->pnh',W,X,optimize=True)+b[:,None,:])
        out=np.einsum('pnh,ph->pn',z,v)+c[:,None]
        return ((out-y)**2).mean(1)+.0001*(p*p).mean(1)
    fit=fitness(pop)
    if mode in ['GA-BP','GA-PSO-BP']:
        for _ in range(SETTINGS['ga_generations']):
            order=np.argsort(fit);new=[pop[order[0]].copy(),pop[order[1]].copy()]
            def select():
                a,b=rng.choice(P,2,replace=False);return pop[a if fit[a]<fit[b] else b]
            while len(new)<P:
                p1,p2=select(),select();child=p1.copy()
                if rng.random()<SETTINGS['crossover_probability']:
                    a=rng.random(L);child=a*p1+(1-a)*p2
                mask=rng.random(L)<SETTINGS['mutation_probability'];child[mask]+=rng.normal(0,.3,mask.sum())
                new.append(np.clip(child,-3,3))
            pop=np.asarray(new,dtype='float32');fit=fitness(pop)
    if mode in ['PSO-BP','GA-PSO-BP']:
        vel=np.zeros_like(pop);best=pop.copy();bf=fit.copy();gb=best[bf.argmin()].copy()
        for t in range(SETTINGS['pso_generations']):
            w=.9+(.4-.9)*t/(SETTINGS['pso_generations']-1)
            vel=np.clip(w*vel+1.5*rng.random(pop.shape)*(best-pop)+1.5*rng.random(pop.shape)*(gb-pop),-1,1)
            pop=np.clip(pop+.5*vel,-3,3).astype('float32');fit=fitness(pop);take=fit<bf
            best[take]=pop[take];bf[take]=fit[take];gb=best[bf.argmin()].copy()
        v=gb
    else:v=pop[fit.argmin()]
    put_vector(net,v)

def _learn(net,X,y,S,epochs,seed,Xv=None,yv=None,Sv=None):
    opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=.0001)
    gen=torch.Generator().manual_seed(seed);best=np.inf;wait=0;best_ep=epochs;state=None;hist=[]
    for e in range(1,epochs+1):
        net.train();ix=torch.randperm(len(y),generator=gen)
        for j in ix.split(SETTINGS['batch_size']):
            opt.zero_grad();out=net(X[j],S[j] if S is not None else None)
            loss=((out-y[j])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                tl=float(((net(X,S)-y)**2).mean());vl=float(((net(Xv,Sv)-yv)**2).mean())
            hist.append({'epoch':e,'train_loss':tl,'validation_loss':vl})
            if vl<best-1e-5:best=vl;wait=0;best_ep=e;state=copy.deepcopy(net.state_dict())
            else:wait+=1
            if wait>=SETTINGS['patience']:break
    if state is not None:net.load_state_dict(state)
    return best_ep,hist

class Predictor:
    def predict_members(self,X,S=None):
        X=np.asarray(X,dtype=float)
        if self.kind=='tree':return np.array([self.est.predict(X)])
        xx=torch.tensor(self.scale.x(X));ss=torch.tensor(self.scale.seq(S)) if self.sequence else None
        out=[]
        for m in self.nets:
            m.eval()
            with torch.no_grad():
                pp=[]
                for j in range(0,len(xx),4096):pp.append(m(xx[j:j+4096],ss[j:j+4096] if ss is not None else None).numpy())
            out.append(np.concatenate(pp)*self.scale.ys+self.scale.ym)
        return np.asarray(out)
    def predict(self,X,S=None):return self.predict_members(X,S).mean(0)

def fit_model(name,X,y,S,validation_mask,seed=42,members=None):
    start=time.perf_counter();X=np.asarray(X,float);y=np.asarray(y,float);m=Predictor();m.name=name
    m.feature_count=X.shape[1];m.history=[];m.best_epochs=[]
    if name in ['Ridge','RandomForest','XGBoost']:
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        if name=='Ridge':est=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10))
        elif name=='RandomForest':est=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,n_jobs=2,random_state=seed)
        else:est=XGBRegressor(n_estimators=350,max_depth=4,learning_rate=.04,subsample=.8,colsample_bytree=.8,n_jobs=2,random_state=seed)
        m.kind='tree';m.est=est.fit(X,y);m.parameters=None;m.seconds=time.perf_counter()-start;return m
    m.kind='neural';m.sequence=name=='LSTM';m.nets=[]
    vm=np.asarray(validation_mask,bool).copy();assert vm.any() and (~vm).any()
    m.scale=Scale().fit(X[~vm],y[~vm],S[~vm])
    xx=torch.tensor(m.scale.x(X));yy=torch.tensor(m.scale.y(y));ss=torch.tensor(m.scale.seq(S)) if m.sequence else None
    for z in range(members or SETTINGS['ensemble_seeds']):
        se=seed+z;torch.manual_seed(se);net=Net(X.shape[1],m.sequence)
        if name in ['GA-BP','PSO-BP','GA-PSO-BP']:meta_initialise(net,xx[~vm].numpy(),yy[~vm].numpy(),se,name)
        initial=copy.deepcopy(net.state_dict())
        ep,hist=_learn(net,xx[~vm],yy[~vm],ss[~vm] if ss is not None else None,SETTINGS['epochs_max'],se,
                      xx[vm],yy[vm],ss[vm] if ss is not None else None)
        # Refit on the entire outer training set; test data never select epoch or scaler.
        net.load_state_dict(initial);_learn(net,xx,yy,ss,ep,se)
        m.nets.append(net);m.best_epochs.append(ep)
        if z==0:m.history=hist
    m.parameters=sum(p.numel() for p in m.nets[0].parameters());m.seconds=time.perf_counter()-start
    return m
