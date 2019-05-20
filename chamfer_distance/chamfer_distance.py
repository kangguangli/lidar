import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace as br

from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["chamfer_distance/chamfer_distance.cpp",
                   "chamfer_distance/chamfer_distance.cu"])

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())


        if not gradxyz1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)


# def ChamferLoss(points1, points2):

#     chamfer_dist = ChamferDistance()
#     dist1, dist2 = chamfer_dist(points1, points2)
#     return (torch.mean(dist1)) + (torch.mean(dist2))


class ChamferLoss(nn.Module):

	def __init__(self):
		super(ChamferLoss, self).__init__()
		self.eps = 1e-8

	def forward(self,preds,gts):
		P = self.batch_pairwise_dist(gts, preds)
		br()
		mins, _ = torch.min(P, 1)
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2

	def batch_pairwise_dist(self,x,y):
		bs, num_points, points_dim = x.size()
		#br()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		diag_ind = torch.arange(0, num_points).type(torch.LongTensor)
		rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
		ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
		br()
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P