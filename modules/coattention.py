import torch
import torch.nn as nn
import torch.nn.functional as F

class Coattention(nn.Module):
    def __init__(self, hidden_dim, hidden_dim1):
        super(Coattention, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_proj = nn.Linear(hidden_dim, hidden_dim1)


    def forward(self, Q, D):
        #project d
        D = F.tanh(self.d_proj(D))
        #co attention
        D_t = torch.transpose(D, 1, 2) # d x lT
        L = torch.bmm(Q, D_t) # vT x lT

        A_Q_ = F.softmax(L, dim=1)
        A_Q = torch.transpose(A_Q_, 1, 2) #lT x vT
        C_Q = torch.bmm(D_t, A_Q) #d x vT

        Q_t = torch.transpose(Q, 1, 2) #d x vT
        A_D = F.softmax(L, dim=2)
        C_D = torch.bmm(torch.cat((Q_t, C_Q), 1), A_D)  # 2d x vT x vT x lT= 2d x lT

        C_D_t = torch.transpose(C_D, 1, 2) #lT x 2d

        return C_D_t