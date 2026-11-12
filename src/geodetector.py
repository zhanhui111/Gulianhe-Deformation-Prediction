"""Four-way OPGD search and fixed-cutpoint block bootstrap for every interaction.

Intervals condition on the selected cutpoints (selection uncertainty is not
included). q measures association, not a causal contribution or forecast skill.
"""
import json
import numpy as np
import pandas as pd
import jenkspy
from core import *

def qstat(y,g):
    n=np.bincount(g);s=np.bincount(g,weights=y);s2=np.bincount(g,weights=y*y)
    return 1-(s2-s*s/np.maximum(n,1)).sum()/((y-y.mean())**2).sum()

def cutpoints(x,k,method):
    if method=='equal':cuts=np.linspace(x.min(),x.max(),k+1)
    elif method=='quantile':cuts=np.quantile(x,np.linspace(0,1,k+1))
    elif method=='geometric':
        shift=max(0,1-x.min());cuts=np.geomspace(x.min()+shift,x.max()+shift,k+1)-shift
    else:
        # A fixed unsupervised subsample bounds natural-break search cost.
        r=np.random.default_rng(42);sample=x if len(x)<=4000 else x[r.choice(len(x),4000,replace=False)]
        cuts=np.asarray(jenkspy.jenks_breaks(sample,n_classes=k));cuts[0]=x.min();cuts[-1]=x.max()
    return np.unique(cuts[1:-1])

def main():
    d,dates=load_points();X,_,last,_=features_at(d,dates,pd.Timestamp(CUTOFF[2:]),pd.Timestamp(TARGET_COL[2:]))
    factors=['elev','slope','rough','relief','aspect_sin','aspect_cos','d_last','trend','seasonal_amplitude',
             'thaw_index_365','freeze_index_365','temp_30','precip_90','snow_days_365']
    if 'curvature' in d:X['curvature']=d.curvature;factors.append('curvature')
    y=d[TARGET_COL].to_numpy()-last;b=spatial_splits(d).block.to_numpy();u,bi=np.unique(b,return_inverse=True)
    r=np.random.default_rng(42);weights=r.multinomial(len(u),np.repeat(1/len(u),len(u)),size=1000)
    B=len(u);W=weights.astype(float);groups={};rows=[];excluded=[];cutslog={}
    # Missing curvature, if any, is median imputed for descriptive attribution only.
    X=X.fillna(X.median())
    def bootstrap(g):
        L=g.max()+1;code=bi*L+g
        n=np.bincount(code,minlength=B*L).reshape(B,L);s=np.bincount(code,weights=y,minlength=B*L).reshape(B,L)
        s2=np.bincount(code,weights=y*y,minlength=B*L).reshape(B,L)
        nn=W@n;ss=W@s;ss2=W@s2
        total=ss2.sum(1)-ss.sum(1)**2/nn.sum(1)
        q=1-(ss2-ss*ss/np.maximum(nn,1)).sum(1)/total
        return np.quantile(q,[.025,.975])
    for f in factors:
        x=X[f].to_numpy()
        if len(np.unique(x))<3:
            excluded.append({'factor':f,'reason':'constant or fewer than three distinct values on spatial support'});continue
        best=(-np.inf,None,None,None,None)
        for method in ['equal','natural','quantile','geometric']:
            for k in range(3,min(10,len(np.unique(x)))+1):
                try:cuts=cutpoints(x,k,method);g=np.searchsorted(cuts,x,side='right');q=qstat(y,g)
                except (ValueError,IndexError):continue
                if q>best[0]:best=(q,method,k,g,cuts)
        q,method,k,g,cuts=best;groups[f]=g;lo,hi=bootstrap(g);cutslog[f]=cuts.tolist()
        rows.append({'factor':f,'q':q,'ci_low':lo,'ci_high':hi,'method':method,'requested_classes':k,
                     'occupied_classes':len(np.unique(g)),'n':len(y),'n_blocks':B})
        print('OPGD',f,round(q,4),flush=True)
    save_table(pd.DataFrame(rows).sort_values('q',ascending=False),'opgd_q.csv')
    save_table(pd.DataFrame(excluded,columns=['factor','reason']),'opgd_excluded.csv')
    inter=[];names=list(groups);qd={a['factor']:a['q'] for a in rows}
    for i,a in enumerate(names):
        for c in names[i+1:]:
            pair=np.column_stack([groups[a],groups[c]]);_,g=np.unique(pair,axis=0,return_inverse=True)
            q=qstat(y,g);lo,hi=bootstrap(g)
            typ='nonlinear enhancement' if q>qd[a]+qd[c] else 'bivariate enhancement' if q>max(qd[a],qd[c]) else 'no enhancement'
            inter.append({'factor_a':a,'factor_b':c,'q_joint':q,'ci_low':lo,'ci_high':hi,'type':typ})
    save_table(pd.DataFrame(inter),'opgd_interaction.csv')
    (TAB/'opgd_settings.json').write_text(json.dumps({'response':'2021-12-14 LOS minus 2020-11-01 LOS',
      'bootstrap':'1000 resamples of 500 m blocks; cutpoints held fixed','natural_breaks':'Jenks on fixed random sample of at most 4000 factor values',
      'selection_limitation':'Intervals condition on discretisation; target-guided maximum-q selection can be optimistic.',
      'climate_limitation':'One coarse regional grid; temperature indices vary spatially only through elevation correction. They are not independent measured spatial thermal fields.',
      'cutpoints':cutslog},indent=2),encoding='utf-8')

if __name__=='__main__':main()
