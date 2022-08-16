import torch
import math
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from ctx import ctx

device = ctx["device"]


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(-1)

class MPToEdge(MessagePassing):
    def __init__(self, in_x, in_w, out_w, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_x = in_x
        self.in_w = in_w
        self.out_w = out_w

        self.lin_x = torch.nn.Linear(in_x, out_w, bias=bias)
        self.lin_w = torch.nn.Linear(in_w, out_w, bias=bias)

        self.relu = torch.nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_x.reset_parameters()
        self.lin_w.reset_parameters()

    def forward(self, h, x, w):
        x = self.lin_x(x)

        w = self.lin_w(w)

        size = (x.shape[-2], w.shape[-2])

        return self.propagate(h, x=x, w=w, size=size) # W

    def message(self, x_j=None):
        return x_j

    def update(self, aggr_out, w=None):
        return w + aggr_out * w


class MPToVertex(MessagePassing):
    def __init__(self, in_x, in_w, out_x, aggr='add', bias=True, **kwargs):
        super().__init__(aggr=aggr, flow='target_to_source', **kwargs)

        self.in_x = in_x
        self.in_w = in_w
        self.out_x = out_x

        self.lin_x = torch.nn.Linear(in_x, out_x, bias=bias)
        self.lin_w = torch.nn.Linear(in_w, out_x, bias=bias)

        self.relu = torch.nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_x.reset_parameters()

        self.lin_w.reset_parameters()

    def forward(self, h, x, w):
        x = self.lin_x(x)

        w = self.lin_w(w)

        size = (w.shape[-2], x.shape[-2])

        return self.propagate(h, x=x, w=w, size=size) # X

    def message(self, w_j=None):
        return w_j

    def update(self, aggr_out, x=None):
        return x + aggr_out * x


class HyperEConv(torch.nn.Module):
    def __init__(self, in_x, in_w, out_w, out_x, reverse=False, normalize_w=False, normalize_x=False, **kwargs):
        super().__init__(**kwargs)

        self.reverse = reverse

        if self.reverse:
            self.mte = MPToEdge(in_x, in_w, out_w)
            self.mtx = MPToVertex(in_x, out_w, out_x)
        else:
            self.mtx = MPToVertex(in_x, in_w, out_x)
            self.mte = MPToEdge(out_x, in_w, out_w)

        self.normalize_w = normalize_w
        self.normalize_x = normalize_x

        if self.normalize_w:
            self.nw = torch.nn.BatchNorm1d(out_w)
        if self.normalize_x:
            self.nx = torch.nn.BatchNorm1d(out_x)

    def reset_parameters(self):
        self.mtx.reset_parameters()
        self.mte.reset_parameters()

    def forward(self, h, x, w):
        if self.reverse:
            w = self.mte(h, x, w)
            if self.normalize_w:
                w = self.nw(w)
            x = self.mtx(h, x, w)
            if self.normalize_x:
                x = self.nx(x)
        else:
            x = self.mtx(h, x, w)
            if self.normalize_x:
                x = self.nx(x)
            w = self.mte(h, x, w)
            if self.normalize_w:
                w = self.nw(w)

        return w, x


class MPNet(torch.nn.Module):
    def __init__(self, nodes, edges, F, G, similarities, repeat=1, mode=0, normalize_conv_w=False, normalize_conv_x=False):
        super(MPNet, self).__init__()
        self.mode = mode
        self.x = G
        self.w = F
        self.g2_c3_repeat = repeat
        self.normalize_conv_w = normalize_conv_w
        self.normalize_conv_x = normalize_conv_x

        self.sim_mat = torch.stack([torch.Tensor().to(device) for i in range(similarities)])

        modifier = 2

        self.c1 = self.add_hypere(modifier*4, modifier*4)
        self.c2 = self.add_hypere(modifier*2, modifier*2)
        self.output = self.create_output(nodes, edges, similarities)
        self.activation = torch.nn.Sigmoid()

    def create_output(self, nodes, edges, similarities):
        print(f"Creating output with {nodes} and {edges}")
        return torch.nn.Sequential(
            torch.nn.Linear(similarities*3, 1),
            torch.nn.Sigmoid(),
            Flatten(),
        )

    def forward(self, sample):
        x = sample.x
        h = sample.h
        w = sample.w

        w, x = self.forward_hypere(self.c1, h, x, w)
        w, x = self.forward_hypere(self.c2, h, x, w)

        if self.mode == 2:
            return x, w

        if self.mode == 3:
            return x, w
        if self.mode == 1:
            return x, w

        if self.mode == 10:
            backx = x
            backw = w

        if hasattr(sample, "w_batch"):
            w_batch_mask = sample.w_batch
            x_batch_mask = sample.x_batch
        else:
            w_batch_mask = None
            x_batch_mask = None

        x, xmask = to_dense_batch(x, x_batch_mask)
        w, wmask = to_dense_batch(w, w_batch_mask)

        x = w

        x = x.flatten(start_dim=1)

        if self.mode == -1:
            return x

        x = self.do_similarities(x)

        x = self.output(x)

        if self.mode == 10:
            return x, backx, backw

        return x

    def add_hypere(self, out_w, out_x):
        tmp = HyperEConv(self.x, self.w, out_w, out_x, normalize_w=self.normalize_conv_w, normalize_x=self.normalize_conv_x)
        if "mte_activation" in ctx:
            tmp.mte.relu = ctx["mte_activation"]()
        if "mtx_activation" in ctx:
            tmp.mtx.relu = ctx["mtx_activation"]()
        self.x = out_x
        self.w = out_w
        return tmp

    def forward_hypere(self, c, h, x, w):
        w, x = c(h, x, w)
        x = self.activation(x)
        w = self.activation(w)
        return w, x

    def set_comparisons(self, X):
        bak_mode = self.mode

        self.mode = -1
        self.sim_mat= torch.stack([torch.Tensor(self(i)[-1].detach().cpu().numpy()).to(device) for i in X])
        self.reset_parameters()

        self.mode = bak_mode

    def do_similarities(self, x):
        window_size = x.shape[-1]
        for i in range(len(self.sim_mat)):
            window_size = math.gcd(window_size, self.sim_mat[i].shape[-1])

        out = []
        splits = x.unfold(-1, window_size, 1)
        for i in range(len(self.sim_mat)):
            diffs = F.cosine_similarity(splits, self.sim_mat[i], -1)

            out.append(diffs.sum(dim=-1)/len(diffs))
            out.append(diffs.min(dim=-1)[0])
            out.append(diffs.max(dim=-1)[0])
        out = torch.stack(out).T
        return out
