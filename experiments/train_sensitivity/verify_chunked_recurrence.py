"""Check outputs and parameter derivatives of the memory-saving recurrence."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.heads.dense_heads import ChunkedPrimary
from src.models.lpu_stream import LPUStreamModel
from src.evaluation.protocol import dump

def main():
    torch.set_num_threads(2);torch.manual_seed(20261030)
    full=LPUStreamModel(quantiles=[.05,.5,.95],enforce_monotonic=True).eval()
    chunk=ChunkedPrimary(quantiles=[.05,.5,.95],enforce_monotonic=True).eval();chunk.load_state_dict(full.state_dict())
    x=torch.randn(3,365,15);s=torch.randn(3,13)
    a=full(x,s);b=chunk(x,s);outdiff=float((a-b).abs().max())
    a.square().mean().backward();b.square().mean().backward()
    diffs={name:float((p.grad-dict(chunk.named_parameters())[name].grad).abs().max()) for name,p in full.named_parameters()}
    assert outdiff<1e-6 and max(diffs.values())<2e-6,(outdiff,diffs)
    dump(ROOT/'results/qa/chunked_recurrence_verification.json',{'passed':True,'output_max_abs_difference':outdiff,
         'parameter_gradient_max_abs_differences':diffs,'scope':'365-step CPU forward/backward with identical weights; no dropout inside the recurrence'})
    print('Chunked recurrence equivalence passed',outdiff,max(diffs.values()),flush=True)

if __name__=='__main__':main()
