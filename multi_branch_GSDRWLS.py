from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def random_walk_compute(p_g_score, g_g_score, alpha, walkstep):
    # Random Walk Computation
    one_diag = Variable(torch.eye(g_g_score.size(0)).cuda(), requires_grad=False) 
    g_g_score_sm = Variable(g_g_score.data.clone(), requires_grad=False) 

    if walkstep == 'combine':
        inf_diag_1 = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,1].squeeze().data 
        inf_diag_0 = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,0].squeeze().data 
        A_1 = F.softmax(Variable(inf_diag_1), dim=1) 
        A_0 = F.softmax(Variable(inf_diag_0), dim=1) 
        A_analysis_1 = (1 - alpha) * torch.inverse(one_diag - alpha * A_1)
        A_analysis_0 = (1 - alpha) * torch.inverse(one_diag - alpha * A_0)
        A_1 = A_1.transpose(0, 1)
        A_0 = A_0.transpose(0, 1)
        s_score_pg = torch.Tensor(p_g_score.size())
        s_score_pg[:, :,0] = alpha * torch.matmul(p_g_score[:, :,0], A_0).contiguous()+ (1 - alpha)* p_g_score[:, :,0] 
        s_score_pg[:, :,1] = alpha * torch.matmul(p_g_score[:, :,1], A_1).contiguous()+ (1 - alpha)* p_g_score[:, :,1]
        A_analysis_1 = A_analysis_1.transpose(0, 1)
        A_analysis_0 = A_analysis_0.transpose(0, 1)
        s_score_pg_analysis = torch.Tensor(p_g_score.size())
        s_score_pg_analysis[:, :,0] = torch.matmul(p_g_score[:, :,0], A_analysis_0).contiguous()
        s_score_pg_analysis[:, :,1] = torch.matmul(p_g_score[:, :,1], A_analysis_1).contiguous()
        s_score_pg = s_score_pg.view(-1, 2)
        s_score_pg_analysis = s_score_pg_analysis.view(-1, 2)
        outputs = torch.cat((s_score_pg, s_score_pg_analysis), 0)
        outputs = outputs.contiguous()
    elif walkstep == 'one':
        inf_diag_1 = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,1].squeeze().data 
        inf_diag_0 = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,0].squeeze().data 
        A_1 = F.softmax(Variable(inf_diag_1), dim=1) 
        A_0 = F.softmax(Variable(inf_diag_0), dim=1) 
        A_1 = A_1.transpose(0, 1)
        A_0 = A_0.transpose(0, 1)
        s_score_pg = torch.Tensor(p_g_score.size())
        s_score_pg[:, :,0] = alpha * torch.matmul(p_g_score[:, :,0], A_0).contiguous() + (1 - alpha)* p_g_score[:, :,0]
        s_score_pg[:, :,1] = alpha * torch.matmul(p_g_score[:, :,1], A_1).contiguous() + (1 - alpha)* p_g_score[:, :,1]
        s_score_pg = s_score_pg.view(-1, 2) 
        outputs = s_score_pg.contiguous()
    elif walkstep == 'analysis':
        inf_diag_1 = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,1].squeeze().data 
        inf_diag_0 = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,0].squeeze().data 
        A_1 = F.softmax(Variable(inf_diag_1), dim=1) 
        A_0 = F.softmax(Variable(inf_diag_0), dim=1) 
        A_1 = (1 - alpha) * torch.inverse(one_diag - alpha * A_1)
        A_0 = (1 - alpha) * torch.inverse(one_diag - alpha * A_0)
        A_1 = A_1.transpose(0, 1)
        A_0 = A_0.transpose(0, 1)
        s_score_pg = torch.Tensor(p_g_score.size())
        s_score_pg[:, :,0] = torch.matmul(p_g_score[:, :,0], A_0).contiguous()
        s_score_pg[:, :,1] = torch.matmul(p_g_score[:, :,1], A_1).contiguous()
        s_score_pg = s_score_pg.view(-1, 2) 
        outputs = s_score_pg.contiguous()  
    else:
        print('expected walk step should be one, '
              'analysis, combine '
              'but got {}'.format(walkstep))
    return outputs


class GSDRWLSNet(nn.Module):
    def __init__(self, instances_num=4, base_model=None, embed_model=None, alpha=0.1, walkstep='combine'):
        super(GSDRWLSNet, self).__init__()
        self.instances_num = instances_num
        self.alpha = alpha
        self.base = base_model
        self.embed = embed_model
        self.walkstep = walkstep
        if isinstance(embed_model,torch.nn.DataParallel):
            embed_model = embed_model.module
        for i in range(len(embed_model)):
            setattr(self, 'embed_'+str(i), embed_model[i])

    def forward(self, x):
        x = self.base(x) 
        count = 2048 / (len(self.embed)) 
        outputs = [] 
              
        for j in range(len(self.embed)):
            s_score = self.embed[j](x[:,int(j*count):int((j+1)*count)].contiguous(),                                   
                                      x[:,int(j*count):int((j+1)*count)].contiguous())

            for i in range(len(self.embed)):
                w_score = self.embed[i](x[:,int(i*count):int((i+1)*count)].contiguous(),
                                          x[:,int(i*count):int((i+1)*count)].contiguous())
                outputs.append(random_walk_compute(s_score, w_score, self.alpha, self.walkstep))

        outputs = torch.cat(outputs, 0)
        return outputs
