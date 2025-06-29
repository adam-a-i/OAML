from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn

def build_head(head_type,
               embedding_size,
               class_num,
               m,
               t_alpha,
               h,
               s,
               ):

    if head_type == 'adaface':
        head = AdaFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       h=h,
                       s=s,
                       t_alpha=t_alpha,
                       )
    elif head_type == 'arcface':
        head = ArcFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    elif head_type == 'cosface':
        head = CosFace(embedding_size=embedding_size,
                       classnum=class_num,
                       m=m,
                       s=s,
                       )
    else:
        raise ValueError('not a correct head type', head_type)
    return head

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)
        print(f"[DEBUG] AdaFace kernel id (init): {id(self.kernel)}")

    def forward(self, embeddings, norms, labels):
        # Kernel normalization
        kernel_norm = l2_norm(self.kernel, axis=0)
        
        # Cosine similarity computation
        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        if self.training:
            # Norm processing for stability
            safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
            safe_norms = safe_norms.clone().detach()

            # Update batch statistics
            with torch.no_grad():
                mean = safe_norms.mean().detach()
                std = safe_norms.std().detach()
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

            # Margin scaler computation
            margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
            margin_scaler = margin_scaler * self.h
            margin_scaler = torch.clip(margin_scaler, -1, 1)

            # Angular margin (g_angular)
            m_arc = torch.zeros(labels.size()[0], cosine.size()[1], device=cosine.device)
            m_arc.scatter_(1, labels.reshape(-1, 1), 1.0)
            g_angular = self.m * margin_scaler * -1
            m_arc = m_arc * g_angular
            theta = cosine.acos()
            theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
            cosine = theta_m.cos()

            # Additive margin (g_additive)
            m_cos = torch.zeros(labels.size()[0], cosine.size()[1], device=cosine.device)
            m_cos.scatter_(1, labels.reshape(-1, 1), 1.0)
            g_add = self.m + (self.m * margin_scaler)
            m_cos = m_cos * g_add
            cosine = cosine - m_cos

            # Final scaling
            scaled_cosine_m = cosine * self.s
            
            # Check for bad gradients
            bad_grad = torch.isnan(scaled_cosine_m) | torch.isinf(scaled_cosine_m)
            if bad_grad.any():
                return scaled_cosine_m, bad_grad
            
            return scaled_cosine_m
        else:
            # During evaluation, use fixed margin
            m_hot = torch.zeros(labels.size()[0], cosine.size()[1], device=cosine.device)
            m_hot.scatter_(1, labels.reshape(-1, 1), self.m)

            theta = cosine.acos()
            theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
            cosine_m = theta_m.cos()
            scaled_cosine_m = cosine_m * self.s

            return scaled_cosine_m

class CosFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.4):
        super(CosFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.4
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4

        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class ArcFace(Module):

    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369

        self.eps = 1e-4

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m