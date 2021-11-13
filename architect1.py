import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, teacher, assistant, student, args, device):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.teacher = teacher
        self.assistant = assistant
        self.student = student
        self.lambda_par = args.lambda_par
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.teacher.A(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def critere(self, pred, true, data_count, reduction='mean'):
        reweighting = torch.softmax(self.teacher.architect_param123[data_count:data_count + pred.shape[0]], dim=0) ** 0.5
        if reduction != 'mean':
            crit = nn.MSELoss(reduction=reduction)
            return crit(pred * reweighting, true * reweighting).mean(dim=-1)
        return self.criterion(pred * reweighting, true * reweighting)

    def _compute_unrolled_model(self, input_data, eta, teacher_optimizer, data_count):
        pred, true = self._process_one_batch(input_data, self.teacher)
        loss = self.critere(pred, true, data_count)
        theta = _concat(self.teacher.parameters()).data
        try:
            moment = _concat(teacher_optimizer.state[v]['exp_avg'] for v in self.teacher.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.teacher.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))  # todo : *?
        return unrolled_model

    def _compute_unrolled_model1(self, input_data, eta, unrolled_model, unl_data, assistant_optimizer):
        pred, true = self._process_one_batch(input_data, self.assistant)
        loss1 = self.criterion(pred, true)
        l1, t1 = self._process_one_batch(unl_data, unrolled_model)
        logits1, true1 = self._process_one_batch(unl_data, self.assistant)
        loss2 = self.criterion(logits1, l1)
        loss = loss1 + (self.lambda_par * loss2)
        theta = _concat(self.assistant.W()).data
        try:
            moment = _concat(assistant_optimizer.state[v]['exp_avg'] for v in self.assistant.W()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(loss, self.assistant.W())).data + self.network_weight_decay * theta
        unrolled_assistant = self._construct_model_from_theta1(theta.sub(eta, moment + dtheta))
        return unrolled_assistant

    def _compute_unrolled_model2(self, input_data, eta, unrolled_assistant, unl_data, student_optimizer):
        pred, true = self._process_one_batch(input_data, self.student)
        loss1 = self.criterion(pred, true)
        l1, true = self._process_one_batch(unl_data, unrolled_assistant)
        logits1, true = self._process_one_batch(unl_data, self.student)
        loss2 = self.criterion(logits1, l1)
        loss = loss1 + (self.lambda_par * loss2)
        theta = _concat(self.student.W()).data
        try:
            moment = _concat(student_optimizer.state[v]['exp_avg'] for v in self.student.W()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.student.W())).data + self.network_weight_decay * theta
        unrolled_student = self._construct_model_from_theta2(theta.sub(eta, moment + dtheta))
        return unrolled_student

    def step(self, trn_data, val_data, unl_data, eta, teacher_optimizer,
             unrolled, data_count):
        self.optimizer.zero_grad()
        if unrolled:
            implicit_grads = self._backward_step_unrolled(trn_data, val_data, eta, teacher_optimizer, data_count)
            return implicit_grads
        else:
            self._backward_step(val_data, data_count)

    def step1(self, trn_data, val_data, unl_data, eta, teacher_optimizer,
              assistant_optimizer, unrolled, data_count):
        self.optimizer.zero_grad()

        unrolled_model = self._compute_unrolled_model(trn_data, eta, teacher_optimizer, data_count)
        unrolled_student = self._compute_unrolled_model1(trn_data, eta, unrolled_model,
                                                         unl_data, assistant_optimizer)
        pred, true = self._process_one_batch(val_data, unrolled_student)
        unrolled_stud_loss = self.criterion(pred, true)
        unrolled_stud_loss.backward()

        vector_s_dash = [v.grad.data for v in unrolled_student.W()]

        implicit_grads = self._outer1(vector_s_dash, trn_data, unl_data, unrolled_model, eta, data_count)
        return implicit_grads

    def step2(self, trn_data, val_data, unl_data, eta, teacher_optimizer,
              assistant_optimizer, student_optimizer, unrolled, data_count):
        self.optimizer.zero_grad()

        unrolled_teacher = self._compute_unrolled_model(trn_data, eta, teacher_optimizer, data_count)
        unrolled_assistant = self._compute_unrolled_model1(trn_data, eta, unrolled_teacher,
                                                         unl_data, assistant_optimizer)
        unrolled_student = self._compute_unrolled_model2(trn_data, eta, unrolled_assistant,
                                                          unl_data, student_optimizer)

        pred, true = self._process_one_batch(val_data, unrolled_student)
        unrolled_stud_loss = self.criterion(pred, true)
        unrolled_stud_loss.backward()

        vector_l_dash = [v.grad.data for v in unrolled_student.W()]

        implicit_grads = self._outer2(vector_l_dash, trn_data, unl_data, unrolled_teacher,
                                      unrolled_assistant, eta, data_count)
        return implicit_grads

    def step_all3(self, trn_data, val_data, unl_data, eta, teacher_optimizer,
                  assistant_optimizer, student_optimizer, unrolled, data_count, print_grad=False):
        self.optimizer.zero_grad()

        ig1 = self.step(trn_data, val_data, unl_data, eta, teacher_optimizer,
                        unrolled, data_count)
        ig2 = self.step1(trn_data, val_data, unl_data, eta, teacher_optimizer,
                         assistant_optimizer, unrolled, data_count)
        ig3 = self.step2(trn_data, val_data, unl_data, eta, teacher_optimizer,
                         assistant_optimizer, student_optimizer, unrolled, data_count)
        if print_grad:
            G1, G2, G3 = [], [], []
            for g1, g2, g3 in zip(ig1, ig2, ig3):
                G1.append(g1.norm())
                G2.append(g2.norm())
                G3.append(g3.norm())
            print('ig1\n', G1, '\nig2\n', G2, '\nig3', G3)
        implicit_grads = [(x + y + z) for x, y, z in zip(ig1, ig2, ig3)]

        for v, g in zip(self.teacher.A(), implicit_grads):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()
        return implicit_grads

    def _backward_step(self, val_data, data_count):
        pred, true = self._process_one_batch(val_data, self.teacher)
        loss = self.critere(pred, true, data_count)
        loss.backward()

    def _backward_step_unrolled(self, trn_data, val_data, eta, teacher_optimizer, data_count):
        unrolled_model = self._compute_unrolled_model(trn_data, eta, teacher_optimizer, data_count)
        pred, true = self._process_one_batch(val_data, unrolled_model)
        unrolled_loss = self.criterion(pred, true)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.A()]
        vector = [v.grad.data for v in unrolled_model.W()]
        implicit_grads = self._hessian_vector_product(vector, trn_data, data_count)
        return implicit_grads

    def _construct_model_from_theta(self, theta):
        model_new = self.teacher.new()
        model_dict = self.teacher.state_dict()

        params, offset = {}, 0
        for k, v in self.teacher.named_parameters():
            v_length = np.prod(v.size())
            if 'architect_param123' in k:
                params[k] = v
            else:
                params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta1(self, theta):
        model_new = self.assistant.new()
        model_dict = self.assistant.state_dict()

        params, offset = {}, 0
        for k, v in self.assistant.named_W():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta2(self, theta):
        model_new = self.student.new()
        model_dict = self.student.state_dict()

        params, offset = {}, 0
        for k, v in self.student.named_W():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input_data, data_count, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.teacher.W(), vector):
            p.data.add_(R, v)
        pred, true = self._process_one_batch(input_data, self.teacher)
        loss = self.critere(pred, true, data_count)
        grads_p = torch.autograd.grad(loss, self.teacher.A())

        for p, v in zip(self.teacher.W(), vector):
            p.data.sub_(2 * R, v)
        pred, true = self._process_one_batch(input_data, self.teacher)
        loss = self.critere(pred, true, data_count)
        grads_n = torch.autograd.grad(loss, self.teacher.A())

        for p, v in zip(self.teacher.W(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _outer1(self, vector_s_dash, trn_data, unl_data, unrolled_model, eta, data_count, r=1e-2):
        R1 = r / _concat(vector_s_dash).norm()
        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.add_(R1, v)
        logits1, _ = self._process_one_batch(unl_data, self.assistant)
        logits2, _ = self._process_one_batch(unl_data, unrolled_model)
        loss1 = self.criterion(logits1, logits2)

        vector_t_dash = torch.autograd.grad(loss1, unrolled_model.W())
        grad_part1 = self._hessian_vector_product(vector_t_dash, trn_data, data_count, r)

        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.sub_(2 * R1, v)
        logits1, _ = self._process_one_batch(unl_data, self.assistant)
        logits2, _ = self._process_one_batch(unl_data, unrolled_model)
        loss2 = self.criterion(logits1, logits2)

        vector_t_dash = torch.autograd.grad(loss2, unrolled_model.W())
        grad_part2 = self._hessian_vector_product (vector_t_dash, trn_data, data_count, r)

        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.add_(R1, v)

        return [(x - y).div_((2 * R1) / (eta * eta * self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]

    def _outer2(self, vector_l_dash, trn_data, unl_data, unrolled_model, unrolled_assistant, eta, data_count,
                r=1e-2):
        R2 = r / _concat(vector_l_dash).norm()
        for p, v in zip(self.student.W(), vector_l_dash):
            p.data.add_(R2, v)
        logits1, _ = self._process_one_batch(unl_data, self.student)
        logits2, _ = self._process_one_batch(unl_data, unrolled_assistant)
        loss1 = self.criterion(logits1, logits2)

        vector_s_dash = torch.autograd.grad(loss1, unrolled_assistant.W())
        R1 = r / _concat(vector_s_dash).norm()
        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.add_(R1, v)
        logits1, _ = self._process_one_batch(unl_data, self.assistant)
        logits2, _ = self._process_one_batch(unl_data, unrolled_model)
        loss1 = self.criterion(logits1, logits2)

        vector_t_dash = torch.autograd.grad(loss1, unrolled_model.W())
        grad_part11 = self._hessian_vector_product(vector_t_dash, trn_data, data_count, r)

        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.sub_(2 * R1, v)
        logits1, _ = self._process_one_batch(unl_data, self.assistant)
        logits2, _ = self._process_one_batch(unl_data, unrolled_model)
        loss2 = self.criterion(logits1, logits2)

        vector_t_dash = torch.autograd.grad(loss2, unrolled_model.W())
        grad_part12 = self._hessian_vector_product(vector_t_dash, trn_data, data_count, r)

        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.add_(R1, v)

        grad_part1 = [(x - y).div_((2 * R1) / (eta * eta * self.lambda_par)) for x, y in zip(grad_part11, grad_part12)]
        # grad_part1 = self._outer3(self, vector_s_dash, input_train, target_train, input_unlabeled, unrolled_model, eta, R1)
        ############################################################################################################################
        for p, v in zip(self.student.W(), vector_l_dash):
            p.data.sub_(2 * R2, v)
        logits1, _ = self._process_one_batch(unl_data, self.student)
        logits2, _ = self._process_one_batch(unl_data, unrolled_assistant)
        loss2 = self.criterion(logits1, logits2)

        vector_s_dash = torch.autograd.grad(loss2, unrolled_assistant.W())
        R1 = r / _concat(vector_s_dash).norm()
        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.add_(R1, v)
        logits1, _ = self._process_one_batch(unl_data, self.assistant)
        logits2, _ = self._process_one_batch(unl_data, unrolled_model)
        loss1 = self.criterion(logits1, logits2)

        vector_t_dash = torch.autograd.grad(loss1, unrolled_model.W())
        grad_part21 = self._hessian_vector_product(vector_t_dash, trn_data, data_count, r)

        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.sub_(2 * R1, v)
        logits1, _ = self._process_one_batch(unl_data, self.assistant)
        logits2, _ = self._process_one_batch(unl_data, unrolled_model)
        loss2 = self.criterion(logits1, logits2)

        vector_t_dash = torch.autograd.grad(loss2, unrolled_model.W())
        grad_part22 = self._hessian_vector_product(vector_t_dash, trn_data, data_count, r)

        for p, v in zip(self.assistant.W(), vector_s_dash):
            p.data.add_(R1, v)

        grad_part2 = [(x - y).div_((2 * R1) / (eta * eta * self.lambda_par)) for x, y in zip(grad_part21, grad_part22)]

        for p, v in zip(self.student.W(), vector_l_dash):
            p.data.add_(R2, v)

        return [(x - y).div_((2 * R2) / (eta * eta * self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]

    def _process_one_batch(self, data, model):
        batch_x = data[0].float().to(self.device)
        batch_y = data[1].float().to(self.device)

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
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = self.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
