""" Architect controls architecture of cell by computing gradients of alphas """
import copy

import torch
import torch.nn as nn
from utils.tools import broadcast_coalesced, all_reduce_coalesced
import torch.distributed as dist
import utils.tools as tools

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, device, args, criterion, inverse_transform=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.device = device
        self.args = copy.deepcopy(args)
        self.criterion = criterion
        if type(net) == nn.parallel.DistributedDataParallel:
            self.net_in = net
            self.net = self.net_in.module
            self.v_net = copy.deepcopy(net)
        else:
            self.net = net.to(self.device)
            self.v_net = copy.deepcopy(net)
        self.w_momentum =self.args.w_momentum
        self.w_weight_decay = self.args.w_weight_decay
        if self.args.inverse:
            self.inverse_transform = inverse_transform

    def critere(self, pred, true, data_count, reduction='mean'):
        return self.criterion(pred * self.net.arch[data_count:data_count + self.args.batch_size] ** 0.5,
                              true * self.net.arch[data_count:data_count + self.args.batch_size] ** 0.5,
                              reduction=reduction)

    def virtual_step(self, trn_data, next_data, xi, w_optim, data_count):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        pred = torch.zeros(trn_data[1][:, -self.args.pred_len:, :].shape).to(self.device)
        if self.args.rank == 0:
            pred, true = self._process_one_batch(trn_data, self.net)
            unreduced_loss = self.critere(pred, true, data_count, reduction='none')
            gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W(), retain_graph=True)
            with torch.no_grad():
                for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                    m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                    vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                    w.grad = g
                for a, va in zip(self.net.A(), self.v_net.A()):
                    va.copy_(a)
        for r in range(0, self.args.world_size-1):
            if self.args.rank == r:
                pred, true = self._process_one_batch(next_data, self.v_net)
            dist.broadcast(pred.contiguous(), r)
            if self.args.rank == r+1:
                trn_data[1] = torch.cat([trn_data[1][:, :self.args.label_len, :], pred], dim=1)
                pred, true = self._process_one_batch(trn_data, self.net)
                unreduced_loss = self.critere(pred, true, data_count, reduction='none')
                gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W())
                with torch.no_grad():
                    for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                        m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                        vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                        w.grad = g
                    for a, va in zip(self.net.A(), self.v_net.A()):
                        va.copy_(a)
        return unreduced_loss

    def unrolled_backward(self, args_in, trn_data, val_data, next_data, xi, w_optim, data_count):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # init config
        args = args_in
        # do virtual step (calc w`)
        unreduced_loss = self.virtual_step(trn_data, next_data, xi, w_optim, data_count)
        hessian = torch.zeros(args.batch_size, args.pred_len, trn_data[1].shape[-1]).to(self.device)
        if self.args.rank == 1:
            # calc unrolled loss
            pred, true = self._process_one_batch(val_data, self.v_net)
            loss = self.criterion(pred, true)

            # compute gradient
            v_W = list(self.v_net.W())
            dw = list(torch.autograd.grad(loss, v_W))

            hessian = self.compute_hessian(dw, trn_data, data_count)
        elif self.args.rank == 0:
            dw_list = []
            for i in range(self.args.batch_size):
                dw_list.append(torch.autograd.grad(unreduced_loss[i], self.net.W(), retain_graph=i!=self.args.batch_size-1))
        dist.all_reduce(hessian)
        da = torch.zeros(self.args.batch_size).to(self.device)
        if self.args.rank == 0:
            pred, true = self._process_one_batch(trn_data, self.v_net)
            pseudo_loss = pred * hessian
            dw0 = torch.autograd.grad(pseudo_loss, self.v_net.W())
            for i in range(self.args.batch_size):
                for a, b in zip(dw_list[i], dw0):
                    da[i] += (a*b).sum()
        dist.broadcast(da, 0)

        # clipping hessian
        # max_norm = float(args.max_hessian_grad_norm)
        # hessian_clip = copy.deepcopy(hessian)
        # for n, (h_c, h) in enumerate(zip(hessian_clip, hessian)):
        #     h_norm = torch.norm(h.detach(), dim=-1)
        #     max_coeff = h_norm / max_norm
        #     max_coeff[max_coeff < 1.0] = torch.tensor(1.0).cuda(args.gpu)
        #     hessian_clip[n] = torch.div(h, max_coeff.unsqueeze(-1))
        # hessian = hessian_clip

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for a, da in zip(self.net.A(), da):
                a.grad = da * xi * xi
        return unreduced_loss.mean()


    def compute_hessian(self, dw, trn_data, data_count):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01  ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        trn_data[1].requires_grad = True

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d
        pred, true = self._process_one_batch(trn_data, self.net)
        loss = self.critere(pred, true, data_count)
        dE_pos = torch.autograd.grad(loss, [trn_data[1][:, -self.args.pred_len, :]])[0]
        # dE_poss = [torch.zeros(dE_pos.shape).to(self.device) for i in range(self.args.world_size)]
        # dist.all_gather(dE_poss, dE_pos)
        # if self.args.rank < self.args.world_size-1:
        #     pred, _ = self._process_one_batch(next_data, self.v_net)
        #     pseudo_loss = (pred*dE_poss[self.args.rank+1]).sum()
        #     dH2_wpos = list(torch.autograd.grad(pseudo_loss, self.v_net.W()))
        #     for i in zero_list2:
        #         dH2_wpos[i] *= 0
        # for i in zero_list:
        #     dH_wpos[i] *= 0


        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p -= 2. * eps_w * d
        pred, true = self._process_one_batch(trn_data, self.net)
        loss = self.critere(pred, true, data_count)
        dE_neg = torch.autograd.grad(loss, [trn_data[1][:, -self.args.pred_len, :]])
        # dD_wnegs = [torch.zeros(dD_wneg.shape).to(self.device) for i in range(args.world_size)]
        # dist.all_gather(dD_wnegs, dD_wneg)
        # if args.rank < args.world_size-1:
        #     pred, _ = self._process_one_batch(next_data, self.v_net)
        #     pseudo_loss = (pred*dD_wnegs[args.rank+1]).sum()
        #     dH2_wneg = list(torch.autograd.grad(pseudo_loss, self.v_net.H()))
        #     for i in zero_list2:
        #         dH2_wneg[i] *= 0
        # for i in zero_list:
        #     dH_wneg[i] *= 0

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d

        hessian = [(p - n) / (2. * eps_w) for p, n in zip(dE_pos, dE_neg)]
        trn_data[1].requires_grad = False
        return hessian

    def _process_one_batch(self, data, model):
        batch_x = data[0].float().to(self.device)
        batch_y = data[1].float()

        batch_x_mark = data[2].float().to(self.device)
        batch_y_mark = data[3].float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        if self.args.inverse:
            outputs = self.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
