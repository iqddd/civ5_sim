"""Independent CPU checkpoint images with coalesced CUDA-to-host transfers."""
from collections import defaultdict
from copy import deepcopy
import torch


def cpu_snapshot(value):
    """Freeze tensor values once, grouping CUDA tensors by device and dtype.

    Each CUDA group is flattened for one D2H transfer, then rebuilt as CPU views.
    CPU tensors and non-tensor metadata are copied independently of live state.
    Checkpoints consequently contain portable CPU tensors instead of CUDA tags.
    """
    groups=defaultdict(list);copies={}
    def collect(item):
        if torch.is_tensor(item):
            if id(item) in copies:return
            copies[id(item)]=None
            if item.device.type=='cpu':copies[id(item)]=item.detach().clone()
            else:groups[(item.device,item.dtype)].append(item)
        elif isinstance(item,dict):
            for child in item.values():collect(child)
        elif isinstance(item,(tuple,list)):
            for child in item:collect(child)
    collect(value)
    for tensors in groups.values():
        flat=torch.cat([t.detach().reshape(-1) for t in tensors]).cpu()
        offset=0
        for tensor in tensors:
            count=tensor.numel()
            copies[id(tensor)]=flat[offset:offset+count].view(tensor.shape)
            offset+=count
    def rebuild(item):
        if torch.is_tensor(item):return copies[id(item)]
        if isinstance(item,dict):
            result=type(item)((key,rebuild(child)) for key,child in item.items())
            if hasattr(item,'_metadata'):result._metadata=deepcopy(item._metadata)
            return result
        if isinstance(item,list):return [rebuild(child) for child in item]
        if isinstance(item,tuple):return tuple(rebuild(child) for child in item)
        return deepcopy(item)
    return rebuild(value)
