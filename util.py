import os
import pickle
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import pdb

	
class WeightedBCE(nn.Module):

	def __init__(self, eps=1e-12, use_gpu=True):
		super(WeightedBCE, self).__init__()
		self.eps = eps
		self.use_gpu = use_gpu

	def forward(self, inputs, targets, weights):
		log_probs_pos = torch.log(inputs + self.eps)
		log_probs_neg = torch.log(1 - inputs + self.eps)
		loss1 = - targets * log_probs_pos
		loss2 = -(1 - targets) * log_probs_neg
		loss3 = loss1 + loss2
		loss4 = loss3.mean(1)
		# print('999999999999999999',loss4)
		loss5 = weights * loss4
		loss = loss5.mean()
		# print('0000000000',loss)

		return loss
