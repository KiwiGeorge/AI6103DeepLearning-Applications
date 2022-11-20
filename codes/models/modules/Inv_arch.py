import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#
from .module_util import initialize_weights_xavier
from .st_lstm import SpatioTemporalLSTMCell
from .layers import RSTB, PatchEmbed, PatchUnEmbed, WindowAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .Subnet_constructor import DenseBlock


class ResidualBlockNoBN(nn.Module):
	def __init__(self, nf=64, model='MIMO-VRN'):
		super(ResidualBlockNoBN, self).__init__()
		self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		# honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
		# but this is how we trained the model in the first place and what we reported in the paper
		if model == 'LSTM-VRN':
			self.relu = nn.ReLU(inplace=True)
		elif model == 'MIMO-VRN':
			self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# initialization
		initialize_weights_xavier([self.conv1, self.conv2], 0.1)

	def forward(self, x):
		identity = x
		out = self.relu(self.conv1(x))
		out = self.conv2(out)
		return identity + out


class InvBlockExp(nn.Module):
	def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
		super(InvBlockExp, self).__init__()

		self.split_len1 = channel_split_num
		self.split_len2 = channel_num - channel_split_num

		self.clamp = clamp
		#
		self.F = subnet_constructor(self.split_len2, self.split_len1)
		self.G = subnet_constructor(self.split_len1, self.split_len2)
		self.H = subnet_constructor(self.split_len1, self.split_len2)

		# self.F = Block(self.split_len2, self.split_len1)
		# self.G = Block(self.split_len1, self.split_len2)
		# self.H = Block(self.split_len1, self.split_len2)

	def forward(self, x, rev=False):
		x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

		if not rev:
			y1 = x1 + self.F(x2)
			self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
			y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
		else:
			self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
			y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
			y1 = x1 - self.F(y2)

		return torch.cat((y1, y2), 1)

	def jacobian(self, x, rev=False):
		if not rev:
			jac = torch.sum(self.s)
		else:
			jac = -torch.sum(self.s)

		return jac / x.shape[0]


class HaarDownsampling(nn.Module):
	def __init__(self, channel_in):
		super(HaarDownsampling, self).__init__()
		self.channel_in = channel_in

		self.haar_weights = torch.ones(4, 1, 2, 2)

		# H
		self.haar_weights[1, 0, 0, 1] = -1
		self.haar_weights[1, 0, 1, 1] = -1

		# V
		self.haar_weights[2, 0, 1, 0] = -1
		self.haar_weights[2, 0, 1, 1] = -1

		# D
		self.haar_weights[3, 0, 1, 0] = -1
		self.haar_weights[3, 0, 0, 1] = -1

		self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
		self.haar_weights = nn.Parameter(self.haar_weights)
		self.haar_weights.requires_grad = False

	def forward(self, x, rev=False):
		if not rev:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(1 / 16.)

			out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
			out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
			return out
		else:
			self.elements = x.shape[1] * x.shape[2] * x.shape[3]
			self.last_jac = self.elements / 4 * np.log(16.)

			out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
			out = torch.transpose(out, 1, 2)
			out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
			return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

	def jacobian(self, x, rev=False):
		return self.last_jac


class InvNN(nn.Module):
	def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
		super(InvNN, self).__init__()

		operations = []

		current_channel = channel_in
		for i in range(down_num):
			b = HaarDownsampling(current_channel)
			operations.append(b)
			current_channel *= 4
			for j in range(block_num[i]):
				b = InvBlockExp(subnet_constructor, current_channel, channel_out)
				operations.append(b)

		self.operations = nn.ModuleList(operations)

	def forward(self, x, rev=False, cal_jacobian=False):
		out = x
		jacobian = 0

		if not rev:
			for op in self.operations:
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)
		else:
			for op in reversed(self.operations):
				out = op.forward(out, rev)
				if cal_jacobian:
					jacobian += op.jacobian(out, rev)

		if cal_jacobian:
			return out, jacobian
		else:
			return out

#
# class PredictiveModuleMIMO(nn.Module):
# 	def __init__(self, channel_in, nf, block_num_rbm=8):
# 		super(PredictiveModuleMIMO, self).__init__()
#
# 		self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
# 		residual_block = []
# 		for i in range(block_num_rbm):
# 			residual_block.append(ResidualBlockNoBN(nf))
# 		self.residual_block = nn.Sequential(*residual_block)
#
# 	def forward(self, x):
# 		x = self.conv_in(x)
# 		return self.residual_block(x)


class PredictiveModuleMIMO(nn.Module):
	def __init__(self, channel_in, nf, block_num_rbm=8):
		super(PredictiveModuleMIMO, self).__init__()

		self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
		depth = 2
		num_heads = 3
		window_size = 6
		self.F = RSTB(dim=nf, input_resolution=(36, 36), depth=depth, num_heads=num_heads, window_size=window_size)
		self.G = RSTB(dim=nf, input_resolution=(36, 36), depth=depth, num_heads=num_heads, window_size=window_size)
		self.H = RSTB(dim=nf, input_resolution=(36, 36), depth=depth, num_heads=num_heads, window_size=window_size)
		self.I = RSTB(dim=nf, input_resolution=(36, 36), depth=depth, num_heads=num_heads, window_size=window_size)

	def forward(self, x):
		out = self.conv_in(x)  # 8 225 36 36
		x_size = out.shape[2:4]
		out = self.F(out, x_size)
		out = self.G(out, x_size)
		out = self.H(out, x_size)
		out = self.I(out, x_size)

		return out


class RelBlock(nn.Module):
	def __init__(self, num_of_frames, nf):
		super(RelBlock, self).__init__()
		self.num_of_frames = num_of_frames
		self.DB_heads = nn.ModuleList()
		self.conv_1x1_heads = nn.ModuleList()
		self.conv_merges = nn.ModuleList()
		self.conv_1x1_tails = nn.ModuleList()
		self.DB_tails = nn.ModuleList()
		for i in range(self.num_of_frames):
			self.DB_heads.append(DenseBlock(3, nf))
			self.conv_1x1_heads.append(nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True))
			self.conv_merges.append(nn.Conv2d(nf * self.num_of_frames, nf, kernel_size=3, stride=1, padding=1, bias=True))
			self.conv_1x1_tails.append(nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True))
			self.DB_tails.append(DenseBlock(nf, 3))

	def forward(self, x):
		b, c, h, w = x.size()
		x = x.view(b, c//3, 3, h, w)
		out_head = []
		for i in range(self.num_of_frames):
			out_i = x[:, i, ...]
			out_i = self.DB_heads[i](out_i)
			out_i = self.conv_1x1_heads[i](out_i)
			out_head.append(out_i)
		out_head = torch.cat(out_head, dim=1)

		out_tail = []
		for i in range(self.num_of_frames):
			out_i = self.conv_merges[i](out_head)
			out_i = self.conv_1x1_tails[i](out_i)
			out_i = self.DB_tails[i](out_i)
			out_tail.append(out_i)
		out_tail = torch.cat(out_tail, dim=1)

		x = x.view(b, c, h, w)

		return x + out_tail * 0.2


class Net(nn.Module):
	def __init__(self, opt, subnet_constructor=None, down_num=2):
		super(Net, self).__init__()

		self.model = opt['model']
		opt_net = opt['network_G']

		if self.model == 'LSTM-VRN':
			self.channel_in = opt_net['in_nc']
			self.channel_out = opt_net['out_nc']
		elif self.model == 'MIMO-VRN':
			self.gop = opt['gop']
			self.channel_in = opt_net['in_nc'] * self.gop
			self.channel_out = opt_net['out_nc'] * self.gop
		else:
			raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

		self.block_num = opt_net['block_num']
		self.block_num_rbm = opt_net['block_num_rbm']
		self.nf = self.channel_in * 4 ** down_num - self.channel_in
		self.irn = InvNN(self.channel_in, self.channel_out, subnet_constructor, self.block_num, down_num)

		self.Rel = RelBlock(self.gop, 64)
		self.Relback = RelBlock(self.gop, 64)

		if self.model == 'MIMO-VRN':
			self.pm = PredictiveModuleMIMO(self.channel_in, self.nf)
		else:
			raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')

	def forward(self, x, rev=False, hs=[], direction='f'):
		if self.model == 'MIMO-VRN':
			if not rev:

				x = self.Rel(x)
				out_y = self.irn(x, rev)

				return out_y
			else:
				y, _ = x


				out_z = self.pm(y)
				out_x = self.irn(torch.cat([y, out_z], dim=1), rev)

				out_x = self.Relback(out_x)

				return out_x, out_z

		else:
			raise Exception('Model should be either LSTM-VRN or MIMO-VRN.')
