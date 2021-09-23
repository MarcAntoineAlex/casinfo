import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def cusloss(inp, tar):
    m = nn.Softmax(1)
    lm = nn.LogSoftmax(1)
    lenn = inp.shape[0]
    inp = lm(inp)
    tar = m(tar)
    out = inp*tar
    ll = (out.sum()*(-1))/lenn
    return ll


class Architect(object):

  def __init__(self, model, student, student2, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.student = student
    self.student2 = student2
    self.lambda_par = args.lambda_par
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def _compute_unrolled_model1(self, input, target, eta, unrolled_model, input_unlabeled, student_optimizer):
    loss1 = self.student._loss(input, target)
    l1 = unrolled_model(input_unlabeled)
    logits1 = self.student(input_unlabeled)
    loss2 = cusloss(logits1, l1.detach())
    loss = loss1+(self.lambda_par*loss2)
    theta = _concat(self.student.parameters()).data
    try:
      moment = _concat(student_optimizer.state[v]['momentum_buffer'] for v in self.student.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.student.parameters())).data + self.network_weight_decay*theta
    unrolled_student = self._construct_model_from_theta1(theta.sub(eta, moment+dtheta))
    return unrolled_student
  def _compute_unrolled_model2(self, input, target, eta, unrolled_student, input_unlabeled, student2_optimizer):
    loss1 = self.student2._loss(input, target)
    l1 = unrolled_student(input_unlabeled)
    logits1 = self.student2(input_unlabeled)
    loss2 = cusloss(logits1, l1.detach())
    loss = loss1+(self.lambda_par*loss2)
    theta = _concat(self.student2.parameters()).data
    try:
      moment = _concat(student2_optimizer.state[v]['momentum_buffer'] for v in self.student2.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.student2.parameters())).data + self.network_weight_decay*theta
    unrolled_student2 = self._construct_model_from_theta2(theta.sub(eta, moment+dtheta))
    return unrolled_student2

  def step(self, input_train, target_train, input_valid, target_valid, input_unlabeled, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()
    
  def step1(self, input_train, target_train, input_valid, target_valid, input_unlabeled, eta, network_optimizer, student_optimizer, unrolled):
    self.optimizer.zero_grad()
    
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_student = self._compute_unrolled_model1(input_train, target_train, eta, unrolled_model, input_unlabeled, student_optimizer)
    
    unrolled_stud_loss = unrolled_student._loss(input_valid, target_valid)
    unrolled_stud_loss.backward()
    
    vector_s_dash = [v.grad.data for v in unrolled_student.parameters()]
    
    implicit_grads = self._outer1(vector_s_dash, input_train, target_train, input_unlabeled, unrolled_model, eta)
    
    for v, g in zip(self.model.arch_parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    
    self.optimizer.step()
    
  def step2(self, input_train, target_train, input_valid, target_valid, input_unlabeled, eta, network_optimizer, student_optimizer, student2_optimizer, unrolled):
    self.optimizer.zero_grad()
    
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_student = self._compute_unrolled_model1(input_train, target_train, eta, unrolled_model, input_unlabeled, student_optimizer)
    unrolled_student2 = self._compute_unrolled_model2(input_train, target_train, eta, unrolled_student, input_unlabeled, student2_optimizer)
    
    unrolled_stud2_loss = unrolled_student2._loss(input_valid, target_valid)
    unrolled_stud2_loss.backward()
    
    vector_l_dash = [v.grad.data for v in unrolled_student2.parameters()]
    
    implicit_grads = self._outer2(vector_l_dash, input_train, target_train, input_unlabeled, unrolled_model, unrolled_student, eta)
    
    for v, g in zip(self.model.arch_parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    
    self.optimizer.step()
    

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _construct_model_from_theta1(self, theta):
    model_new = self.student.new()
    model_dict = self.student.state_dict()

    params, offset = {}, 0
    for k, v in self.student.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _construct_model_from_theta2(self, theta):
    model_new = self.student2.new()
    model_dict = self.student2.state_dict()

    params, offset = {}, 0
    for k, v in self.student2.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def _outer1(self, vector_s_dash, input_train, target_train, input_unlabeled, unrolled_model, eta, r=1e-2):
    R1 = r / _concat(vector_s_dash).norm()
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss1 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss1, unrolled_model.parameters())
    grad_part1 = self._hessian_vector_product(vector_t_dash, input_train, target_train)
    
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.sub_(2*R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss2 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss2, unrolled_model.parameters())
    grad_part2 = self._hessian_vector_product(vector_t_dash, input_train, target_train)

    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)

    return [(x-y).div_((2*R1)/(eta*eta*self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]


  def _outer2(self, vector_l_dash, input_train, target_train, input_unlabeled, unrolled_model,unrolled_student, eta, r=1e-2):
    R2 = r / _concat(vector_l_dash).norm()
    for p, v in zip(self.student2.parameters(), vector_l_dash):
      p.data.add_(R2, v)
    logits1 = self.student2(input_unlabeled)
    logits2 = unrolled_student(input_unlabeled)
    loss1 = cusloss(logits1, logits2)
    
    vector_s_dash = torch.autograd.grad(loss1, unrolled_student.parameters())
    R1 = r / _concat(vector_s_dash).norm()
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss1 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss1, unrolled_model.parameters())
    grad_part11 = self._hessian_vector_product(vector_t_dash, input_train, target_train)
    
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.sub_(2*R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss2 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss2, unrolled_model.parameters())
    grad_part12 = self._hessian_vector_product(vector_t_dash, input_train, target_train)

    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)
    
    grad_part1 = [(x-y).div_((2*R1)/(eta*eta*self.lambda_par)) for x, y in zip(grad_part11, grad_part12)]
    #grad_part1 = self._outer3(self, vector_s_dash, input_train, target_train, input_unlabeled, unrolled_model, eta, R1)
############################################################################################################################    
    for p, v in zip(self.student2.parameters(), vector_l_dash):
      p.data.sub_(2*R2, v)
    logits1 = self.student2(input_unlabeled)
    logits2 = unrolled_student(input_unlabeled)
    loss2 = cusloss(logits1, logits2)
    
    vector_s_dash = torch.autograd.grad(loss2, unrolled_student.parameters())
    R1 = r / _concat(vector_s_dash).norm()
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss1 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss1, unrolled_model.parameters())
    grad_part21 = self._hessian_vector_product(vector_t_dash, input_train, target_train)
    
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.sub_(2*R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss2 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss2, unrolled_model.parameters())
    grad_part22 = self._hessian_vector_product(vector_t_dash, input_train, target_train)

    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)
    
    grad_part2 = [(x-y).div_((2*R1)/(eta*eta*self.lambda_par)) for x, y in zip(grad_part21, grad_part22)]

    for p, v in zip(self.student2.parameters(), vector_l_dash):
      p.data.add_(R2, v)

    return [(x-y).div_((2*R2)/(eta*eta*self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]


