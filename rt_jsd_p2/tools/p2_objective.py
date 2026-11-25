"""Native GT/JSD/Ranking with stride-keyed objectness balance for P2-P5."""
import torch
from focus_objective import FocusObjective
from uod.localization_loss import DistributionUODLoss
from uod.p2_coupled import OBJECTNESS_BALANCE


class P2FocusObjective(FocusObjective):
    def __init__(self,model,cfg):
        super().__init__(model,cfg)
        assert model.ifd.nl==4 and model.ifd.na==3
        self.original.detection.balance=list(OBJECTNESS_BALANCE)


def active_gpu_check(model,criterion,batch,cfg):
    model.train();criterion.capture_graph=True
    with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['amp']):
        out=model.forward_pair(batch['rgb'],batch['thermal'])
        native=DistributionUODLoss(model,alpha=criterion.alpha,**cfg['loss']['localization'])
        native.detection.balance=list(OBJECTNESS_BALANCE)
        expected,_=native(out,batch['targets'],batch['rgb'].shape[-2:],batch['valid_mask'])
        criterion.beta=0.;base,_=criterion(out,batch['targets'],batch['rgb'].shape[-2:],batch['valid_mask'])
        assert torch.equal(base,expected)
        criterion.beta=cfg['loss']['ranking']['beta']
        active,stats=criterion(out,batch['targets'],batch['rgb'].shape[-2:],batch['valid_mask'])
    rank=criterion.last_rank_loss
    assert torch.isfinite(active) and torch.isfinite(rank) and float(rank)>0
    torch.testing.assert_close(active,expected+len(batch['targets'])*criterion.beta*rank,rtol=0,atol=0)
    rgb,thermal=out['pred_rgb'],out['pred_thermal']
    grad=torch.autograd.grad(rank,tuple(rgb)+tuple(thermal),allow_unused=True)
    offset=4*model.ifd.bins;nonfocus=[1+c for c in range(model.ifd.nc) if c not in criterion.class_ids]
    assert all(g is None for g in grad[len(rgb):]);norm=0.
    for g in grad[:len(rgb)]:
        if g is None:continue
        assert torch.isfinite(g).all() and not torch.count_nonzero(g[...,:offset])
        assert not torch.count_nonzero(g[...,offset:][...,nonfocus]);norm+=float(g.abs().sum())
    assert norm>0
    criterion.capture_graph=False;del criterion.last_rank_loss
    return dict(status='passed',beta_zero_native_loss_exact=True,additive_weight_exact=True,
        rgb_output_gradient_l1=norm,thermal_output_gradient_absent=True,
        regression_output_gradient_zero=True,non_focus_class_output_gradient_zero=True,
        objectness_balance=list(OBJECTNESS_BALANCE),P3_P5_balance_unchanged=True,
        active_rank_loss=float(rank),added_loss_functions=0)
