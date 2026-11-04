"""GPU-resident training windows over the primary-record cache.

The whole historical input grid resides on the GPU, but only declared
training target dates enter the loss and normalization fit. Shared by the
sensitivity trainer and the independent cache verification.
"""
import numpy as np
import torch


def fit_normalization(raw, train_start, train_end):
    """Fit the declared statistics without allocating GPU window tensors."""
    dates=raw['dates'];fit=(dates>=np.datetime64(train_start))&(dates<=np.datetime64(train_end))
    dyn=raw['dynamic'];y=raw['target'];valid=(raw['mask']>0)&np.isfinite(y)
    dv=dyn[:,fit,:].reshape(-1,dyn.shape[-1]);tv=y[:,fit][valid[:,fit]]
    return {'fit_start':train_start,'fit_end':train_end,'target_mean':float(tv.mean()),'target_std':float(tv.std()+1e-8),
            'dynamic_mean':np.nanmean(dv,axis=0).tolist(),'dynamic_std':(np.nanstd(dv,axis=0)+1e-8).tolist(),
            'static_mean':np.nanmean(raw['static'],axis=0).tolist(),'static_std':(np.nanstd(raw['static'],axis=0)+1e-8).tolist()}


class Windows:
    def __init__(self,raw,train_start,train_end,device,seq=180):
        self.raw=raw;self.device=device;self.seq=seq
        self.dates=raw['dates'];self.basins=raw['basins'];self.nb=len(self.basins);self.nd=len(self.dates)
        dyn=raw['dynamic'];y=raw['target'];valid=(raw['mask']>0)&np.isfinite(y)
        self.norm=fit_normalization(raw,train_start,train_end)
        dm=self.norm['dynamic_mean'];ds=self.norm['dynamic_std'];tm=self.norm['target_mean'];ts=self.norm['target_std']
        sm=self.norm['static_mean'];ss=self.norm['static_std']
        # Match CamelsDataset's float32 normalization arithmetic.
        dn=dyn.astype(np.float32)
        for c in range(dn.shape[-1]):dn[:,:,c]=(dn[:,:,c]-float(dm[c]))/float(ds[c])
        yn=(y.astype(np.float32)-tm)/ts
        sn=raw['static'].astype(np.float32)
        for c in range(sn.shape[-1]):sn[:,c]=(sn[:,c]-float(sm[c]))/float(ss[c])
        self.dynamic=torch.tensor(np.nan_to_num(dn),device=device)
        self.target=torch.tensor(np.nan_to_num(yn),device=device)
        self.mask=torch.tensor(valid.astype(np.float32),device=device)
        self.static=torch.tensor(np.nan_to_num(sn),device=device)
        self.lags=torch.arange(-seq,0,device=device)
    def indices(self,start,end):
        days=np.flatnonzero((self.dates>=np.datetime64(start))&(self.dates<=np.datetime64(end)))
        assert len(days) and days.min()>=self.seq
        return torch.tensor(np.concatenate([b*self.nd+days for b in range(self.nb)]),device=self.device)
    def batch(self,idx):
        b=torch.div(idx,self.nd,rounding_mode='floor');t=idx%self.nd
        return self.dynamic[b[:,None],t[:,None]+self.lags],self.static[b],self.target[b,t,None],self.mask[b,t,None]
    def metadata(self,idx):
        a=idx.cpu().numpy();return self.basins[a//self.nd],self.dates[a%self.nd]
