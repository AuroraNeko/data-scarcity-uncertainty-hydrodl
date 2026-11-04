"""Resumable, protocol-declared 50-basin sensitivity fits.

The whole historical input grid resides on the GPU, but only declared training
target dates enter the loss and normalization fit. Test data never select a model.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.lpu_stream import LPUStreamModel
from src.models.tcn import TCNModel
from src.data.windows import Windows
from src.heads.dense_heads import ChunkedPrimary
from src.sensitivity import specs
from src.utils import set_seed,sha256_file
from src.sensitivity_identity import training_input_identity, require_input_identity, completed_run_valid

BASE=ROOT/'results'
TAUS3=[0.05,0.5,0.95]
TAUS13=[0.005,0.01,0.025,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975,0.99,0.995]

def atomic(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');temp.replace(path)

class DenseQuantileStream(LPUStreamModel):
    def __init__(self,taus):
        super().__init__(quantiles=taus,enforce_monotonic=False)
        self.mid=taus.index(0.5)
    def forward(self,d,s):
        raw=super().forward(d,s)
        med=raw[:,self.mid:self.mid+1]
        lo=med-torch.flip(torch.cumsum(F.softplus(torch.flip(raw[:,:self.mid],[-1]))*0.2,dim=-1),[-1])
        hi=med+torch.cumsum(F.softplus(raw[:,self.mid+1:])*0.2,dim=-1)
        return torch.cat([lo,med,hi],dim=-1)

class QuantileTCN(TCNModel):
    def __init__(self):
        super().__init__(channels=32,n_blocks=6,dropout=0.3)
        self.head[-1]=nn.Linear(16,3)
    def forward(self,d,s):
        raw=super().forward(d,s)
        return torch.cat([raw[:,1:2]-F.softplus(raw[:,0:1]),raw[:,1:2],
                          raw[:,1:2]+F.softplus(raw[:,2:3])],dim=-1)

def make_model(kind,taus,seq_len=180):
    if kind=='tcn':return QuantileTCN()
    if len(taus)>3:return DenseQuantileStream(taus)
    if seq_len==365:return ChunkedPrimary(quantiles=taus,enforce_monotonic=True)
    return LPUStreamModel(quantiles=taus,enforce_monotonic=True)

def loss_fn(q,y,m,taus):
    e=y-q
    return (torch.maximum(taus*e,(taus-1)*e)*m).sum()/m.sum().clamp_min(1)/q.shape[1]

@torch.no_grad()
def predict(model,windows,idx,batch=512):
    model.eval();values=[]
    for part in idx.split(batch):
        d,s,_,_=windows.batch(part);values.append(model(d,s).cpu().numpy())
    return np.concatenate(values).astype(np.float64)*windows.norm['target_std']+windows.norm['target_mean']

@torch.no_grad()
def validation_loss(model,windows,idx,taus,batch=512):
    model.eval();tot=torch.zeros((),device=windows.device);cnt=torch.zeros_like(tot)
    for part in idx.split(batch):
        d,s,y,m=windows.batch(part);n=m.sum()
        tot+=loss_fn(model(d,s),y,m,taus)*n;cnt+=n
    return float((tot/cnt).item())

def run(spec,benchmark=False):
    identity=training_input_identity(BASE,spec)
    for directory in ['tables','logs','checkpoints/sensitivity','predictions/sensitivity']:
        (BASE/directory).mkdir(parents=True,exist_ok=True)
    tag=spec['id'];result=BASE/f'tables/training_{tag}.json'
    if not benchmark and completed_run_valid(ROOT,BASE,spec,identity):
        print('SKIP validated',tag,flush=True);return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2);set_seed(spec['seed'],deterministic=True)
    # Match the original cuDNN default so a rerun reproduces the archived
    # predictions; float32 tensor storage alone does not disable TF32 kernels.
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    raw=np.load(BASE/'cache/primary_raw.npz')
    windows=Windows(raw,spec['train_start'],spec['train_end'],device,seq=spec.get('seq_len',180))
    if windows.norm!=identity['normalization']:raise RuntimeError('Window normalization changed after input validation')
    train=windows.indices(spec['train_start'],spec['train_end'])
    # A validation target is separate from all training targets and CQR targets.
    validation=windows.indices('1995-10-01','1996-09-30') if spec['validation'] else None
    cal_start='1996-10-01' if spec['validation'] else '1995-10-01'
    cal=windows.indices(cal_start,'2000-09-30');test=windows.indices('2000-10-01','2008-12-31')
    taus_list=TAUS13 if spec['head']=='13tau' else TAUS3
    taus=torch.tensor(taus_list,device=device).reshape(1,-1)
    model=make_model(spec['architecture'],taus_list,spec.get('seq_len',180)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)
    epochs=30;batch_size=512
    updates_per_epoch=math.ceil(len(train)/batch_size)
    updates_per_round=36 if spec['budget']=='equal_updates' else updates_per_epoch
    checkpoint=BASE/f'checkpoints/sensitivity/{tag}.pt';progress=BASE/f'checkpoints/sensitivity/{tag}_resume.pt'
    start_epoch=0;history=[];best=math.inf;best_state=None;steps=0;budget_order=None;budget_cursor=0
    if progress.exists() and not benchmark:
        saved=torch.load(progress,map_location=device,weights_only=False)
        require_input_identity(saved,identity,f'{tag} resume')
        if saved['spec']!=spec:raise RuntimeError('Resume protocol differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_epoch=saved['epoch'];history=saved['history'];best=saved['best'];best_state=saved['best_state'];steps=saved['steps']
        budget_order=saved.get('budget_order');budget_cursor=saved.get('budget_cursor',0)
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if device.type=='cuda' and saved.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
        print('RESUME',tag,start_epoch,flush=True)
    print('START',tag,'targets',len(train),'updates/round',updates_per_round,flush=True)
    started=time.time()
    for epoch in range(start_epoch,epochs):
        model.train();tick=time.time();total=torch.zeros((),device=device);count=torch.zeros_like(total)
        if spec['budget']=='equal_updates':
            # Continue across budget-round boundaries instead of repeatedly
            # sampling only the first 36 batches of a new permutation.
            parts=[]
            for _ in range(updates_per_round):
                if budget_order is None or budget_cursor>=len(train):
                    budget_order=train[torch.randperm(len(train),device=device)];budget_cursor=0
                parts.append(budget_order[budget_cursor:budget_cursor+batch_size]);budget_cursor+=batch_size
        else:
            order=train[torch.randperm(len(train),device=device)]
            parts=order.split(batch_size)
        for ib,idx in enumerate(parts):
            optimizer.zero_grad(set_to_none=True);d,s,y,m=windows.batch(idx)
            loss=loss_fn(model(d,s),y,m,taus);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            n=m.sum();total+=loss.detach()*n;count+=n;steps+=1
            if benchmark and ib==19:
                if device.type=='cuda':torch.cuda.synchronize()
                print('BENCHMARK',tag,'20 batches seconds',time.time()-tick,'reserved_MB',torch.cuda.memory_reserved()/2**20,flush=True)
                return
        train_loss=float((total/count).item());scheduler.step(train_loss)
        val=validation_loss(model,windows,validation,taus) if validation is not None else None
        score=val if val is not None else train_loss
        if score<best:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row={'epoch_or_round':epoch+1,'updates':steps,'train_loss':train_loss,'validation_loss':val,
             'train_pinball_log':train_loss*windows.norm['target_std'],
             'validation_pinball_log':None if val is None else val*windows.norm['target_std'],
             'learning_rate':optimizer.param_groups[0]['lr'],'seconds':time.time()-tick}
        history.append(row);print(tag,json.dumps(row),flush=True)
        state={'spec':spec,'input_identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'epoch':epoch+1,'history':history,'best':best,'best_state':best_state,'steps':steps,
               'budget_order':budget_order,'budget_cursor':budget_cursor,
               'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state() if device.type=='cuda' else None}
        temp=progress.with_suffix('.tmp');torch.save(state,temp);temp.replace(progress)
        atomic(BASE/f'tables/progress_{tag}.json',{'spec':spec,'input_identity':identity,'history':history,'completed':False})
    final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'spec':spec,'input_identity':identity,'final_state':final_state,'validation_selected_state':best_state if validation is not None else None,
                'selection':'minimum_independent_validation_pinball' if validation is not None else 'final_epoch',
                'normalization':windows.norm,'quantiles':taus_list,'history':history},checkpoint)
    arrays={'taus':np.asarray(taus_list),'normalization_json':json.dumps(windows.norm),'input_identity_json':json.dumps(identity)}
    for split,idx in [('calibration',cal),('test',test)]:
        b,dates=windows.metadata(idx);i=idx.cpu().numpy();bb=i//windows.nd;tt=i%windows.nd
        arrays[f'{split}_basin']=b;arrays[f'{split}_dates']=dates
        arrays[f'{split}_y_log']=raw['target'][bb,tt].astype(np.float32)
        arrays[f'{split}_mask']=raw['mask'][bb,tt]
        arrays[f'{split}_final_quantiles_log']=predict(model,windows,idx).astype(np.float32)
    if validation is not None:
        # The final-epoch WY1995-trained fit also emits predictions over the
        # fixed 1996-2000 common calibration pool; the analysis synthesizes the
        # temporal_wy1995 reference from them. The checkpoint is fixed in
        # advance and does not use validation for selection.
        if spec['train_start']=='1994-10-01':
            fullcal=windows.indices('1995-10-01','2000-09-30');i=fullcal.cpu().numpy();bb=i//windows.nd;tt=i%windows.nd
            arrays['fullcal_final_quantiles_log']=predict(model,windows,fullcal).astype(np.float32)
            arrays['fullcal_y_log']=raw['target'][bb,tt].astype(np.float32);arrays['fullcal_mask']=raw['mask'][bb,tt]
            arrays['fullcal_basin'],arrays['fullcal_dates']=windows.metadata(fullcal)
        model.load_state_dict(best_state)
        for split,idx in [('calibration',cal),('test',test)]:
            arrays[f'{split}_selected_quantiles_log']=predict(model,windows,idx).astype(np.float32)
    prediction=BASE/f'predictions/sensitivity/{tag}.npz';np.savez_compressed(prediction,**arrays)
    atomic(result,{'completed':True,'spec':spec,'input_identity':identity,'train_target_count':len(train),'optimizer_updates':steps,
        'n_parameters':sum(p.numel() for p in model.parameters()),'normalization':windows.norm,'history':history,
        'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha256_file(checkpoint),
        'predictions':prediction.relative_to(ROOT).as_posix(),'predictions_sha256':sha256_file(prediction),
        'wall_seconds_this_invocation':time.time()-started,'calibration_start':cal_start,'quantiles':taus_list})
    print('DONE',tag,'seconds',time.time()-started,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--id');parser.add_argument('--benchmark',action='store_true');parser.add_argument('--list',action='store_true')
    args=parser.parse_args();all_specs=specs()
    if args.list:print(json.dumps(all_specs,indent=2));sys.exit()
    selected=[s for s in all_specs if s['id']==args.id]
    if len(selected)!=1:raise ValueError('Specify one declared --id')
    run(selected[0],args.benchmark)
