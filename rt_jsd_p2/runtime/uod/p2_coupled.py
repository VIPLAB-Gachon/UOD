"""P2-only extension of the unchanged Coupled25 anchor-based detector.

Reuse existing stride4 backbone layer2 and stride8 neckP3. Existing P3-P5
backbone/neck operations and predictors are untouched. All new parameters use
a private RNG fork; original3 prediction levels shift from indices0..2 to1..3.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .yolov5_official import OfficialYOLOv5UOD,YOLOv5Stage,ensure_three_channels

OBJECTNESS_BALANCE=(4.,4.,1.,.4)  # P2 new; old P3/P4/P5 stay4/1/.4


class DetailBackbone(YOLOv5Stage):
    def forward(self,x,incoming=None):
        cache=dict(incoming or {})
        for module in self.layers:
            source=module.f
            if source==-1:value=x
            elif isinstance(source,int):value=cache[source]
            else:value=[x if i==-1 else cache[i] for i in source]
            x=module(value)
            if module.i in {2,4,6,10,14,17,20,23}:cache[int(module.i)]=x
        return x,tuple(cache[i] for i in self.output_indices),cache


class DetailFusion(nn.Module):
    def __init__(self,detail_channels,semantic_channels,channels,**factory):
        super().__init__()
        self.detail=nn.Sequential(nn.Conv2d(detail_channels,channels,1,**factory),nn.SiLU())
        self.semantic=nn.Sequential(nn.Conv2d(semantic_channels,channels,1,**factory),nn.SiLU())
        self.refine=nn.Sequential(nn.Conv2d(2*channels,channels,3,padding=1,**factory),nn.SiLU())
    def forward(self,detail,semantic):
        fine=self.detail(detail)
        coarse=F.interpolate(self.semantic(semantic),size=fine.shape[-2:],mode='nearest')
        return self.refine(torch.cat((fine,coarse),1))


def install_p2(model,enabled=True,channels=128,initialization_seed=2026,anchor_scale=.5):
    assert enabled and channels==128 and anchor_scale==.5
    assert model.separate_ife and not model.use_nfe
    head=model.ifd
    assert head.nl==3 and head.na==3 and head.bins==25
    assert list(map(float,head.stride))==[8.,16.,32.]
    assert head.m[0].in_channels==channels
    device=head.support.device;factory=dict(device=device,dtype=head.m[0].weight.dtype)
    devices=[device.index] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(initialization_seed)
        if devices:torch.cuda.manual_seed(initialization_seed)
        c2=model.bfe.layers[2].cv3.conv.out_channels
        model.p2_ife_rgb=DetailFusion(c2,head.m[0].in_channels,channels,**factory)
        model.p2_ife_thermal=copy.deepcopy(model.p2_ife_rgb)
        prediction=copy.deepcopy(head.m[0])
        with torch.no_grad():
            # Same score/class rows;4x as many cells -> objectness prior /4.
            prediction.bias.view(head.na,head.no)[:,4*head.bins]-=math.log(4.)
        old_anchors=head.anchors.detach().clone()
        # Pixel anchors half of P3, stride half of P3 => equal grid-unit anchors.
        p2_anchors=old_anchors[:1]*head.stride[0]*anchor_scale/4.
        head.anchors=torch.cat((p2_anchors,old_anchors),0)
        head.stride=torch.tensor([4.,8.,16.,32.],device=device)
        head.m=nn.ModuleList([prediction,*head.m])
        head.nl=4;head.grid=[torch.empty(0,device=device) for _ in range(4)]
        head.anchor_grid=[torch.empty(0,device=device) for _ in range(4)]
        head.type='P2CoupledDistributionDetect'
        head.np=sum(p.numel() for p in head.parameters())
    model.bfe=DetailBackbone(model.bfe.layers,model.bfe.output_indices)
    model.__class__=P2CoupledUOD
    model.stride=head.stride
    return model


def matched_hyp(cfg,model):
    """Keep baseline effective gains; do not silently shrink them with4 levels."""
    h=dict(cfg['loss']['detection'])
    h['box']*=3/3
    h['cls']*=model.nc/80*3/3
    h['obj']*=(cfg['data']['image_size']/640)**2*3/3
    return h


class P2CoupledUOD(OfficialYOLOv5UOD):
    def _features(self,images,modality=None):
        if modality not in ('rgb','thermal'):raise ValueError('Explicit modality required')
        basic,_,cache=self.bfe(ensure_three_channels(images))
        neck=self.ife_rgb if modality=='rgb' else self.ife_thermal
        fusion=self.p2_ife_rgb if modality=='rgb' else self.p2_ife_thermal
        _,features,_=neck(basic,cache)
        return (fusion(cache[2],features[0]),*features)

    def forward_pair(self,rgb,thermal,*,return_features=False):
        rgb,thermal=ensure_three_channels(rgb),ensure_three_channels(thermal)
        if rgb.shape!=thermal.shape:raise ValueError('Aligned pair dimensions differ')
        B=rgb.shape[0];basic,_,cache=self.bfe(torch.cat((rgb,thermal),0))
        cr=self._slice_cache(cache,0,B);ct=self._slice_cache(cache,B,2*B)
        _,fr,_=self.ife_rgb(basic[:B],cr);_,ft,_=self.ife_thermal(basic[B:],ct)
        fr=(self.p2_ife_rgb(cr[2],fr[0]),*fr);ft=(self.p2_ife_thermal(ct[2],ft[0]),*ft)
        raw=self.ifd([torch.cat((r,t),0) for r,t in zip(fr,ft,strict=True)])
        extra=dict(intrinsic_rgb=fr,intrinsic_thermal=ft) if return_features else {}
        if self.training:
            return dict(pred_rgb=tuple(p[:B] for p in raw),pred_thermal=tuple(p[B:] for p in raw),**extra)
        decoded,levels=raw
        return dict(decoded_rgb=decoded[:B],decoded_thermal=decoded[B:],
            raw_rgb=tuple(p[:B] for p in levels),raw_thermal=tuple(p[B:] for p in levels),**extra)
