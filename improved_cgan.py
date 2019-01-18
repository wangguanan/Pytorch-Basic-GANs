#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np

import os


# 超参数
gpu_id = None
if gpu_id is not None:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
if os.path.exists('improved_cgan_images') is False:
	os.makedirs('improved_cgan_images')
z_dim = 100
batch_size = 64
learning_rate = 0.0002
total_epochs = 200


class Discriminator(nn.Module):
	'''全连接判别器，用于1x28x28的MNIST数据,输出是数据和类别'''
	def __init__(self):
		super(Discriminator, self).__init__()

		layers = []
		# 第一层
		layers.append(nn.Linear(in_features=28*28+10, out_features=512, bias=True))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第二层
		layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 输出层
		layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
		layers.append(nn.Sigmoid())

		self.model = nn.Sequential(*layers)

	def forward(self, x, c):
		x = x.view(x.size(0), -1)
		validity = self.model(torch.cat([x, c], -1))
		return validity

class Generator(nn.Module):
	'''全连接生成器，用于1x28x28的MNIST数据，输入是噪声和类别'''
	def __init__(self, z_dim):
		super(Generator, self).__init__()

		layers = []
		# 第一层
		layers.append(nn.Linear(in_features=z_dim+10, out_features=128))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第二层
		layers.append(nn.Linear(in_features=128, out_features=256))
		layers.append(nn.BatchNorm1d(256, 0.8))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第三层
		layers.append(nn.Linear(in_features=256, out_features=512))
		layers.append(nn.BatchNorm1d(512, 0.8))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 输出层
		layers.append(nn.Linear(in_features=512, out_features=28*28))
		layers.append(nn.Tanh())

		self.model = nn.Sequential(*layers)

	def forward(self, z, c):
		x = self.model(torch.cat([z, c], dim=1))
		x = x.view(-1, 1, 28, 28)
		return x


def one_hot(labels, class_num):
	'''把标签转换成one-hot类型'''
	tmp = torch.FloatTensor(labels.size(0), class_num).zero_()
	one_hot = tmp.scatter_(dim=1, index=torch.LongTensor(labels.view(-1, 1)), value=1)
	return one_hot


# 初始化构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)

# 初始化二值交叉熵损失
bce = torch.nn.BCELoss().to(device)
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.999])
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999])

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#用于生成效果图
# 生成100个one_hot向量，每类10个
fixed_c = torch.FloatTensor(100, 10).zero_()
fixed_c = fixed_c.scatter_(dim=1, index=torch.LongTensor(np.array(np.arange(0,10).tolist()*10).reshape([100, 1])), value=1)
fixed_c = fixed_c.to(device)
# 生成100个随机噪声向量
fixed_z = torch.randn([100, z_dim]).to(device)

# 开始训练，一共训练total_epochs
for epoch in range(total_epochs):

	# 在训练阶段，把生成器设置为训练模式；对应于后面的，在测试阶段，把生成器设置为测试模式
	generator = generator.train()

	# 训练一个epoch
	for i, data in enumerate(dataloader):

		# 加载真实数据
		real_images, real_labels = data
		wrong_labels = []
		for real_label in real_labels:
			tmp = np.arange(10).tolist()
			tmp.remove(float(real_label.data))
			wrong_labels.append(np.random.choice(tmp, 1)[0])
		wrong_labels = torch.LongTensor(wrong_labels)
		#
		real_images = real_images.to(device)
		# 把对应的标签转化成 one-hot 类型
		tmp = torch.FloatTensor(real_labels.size(0), 10).zero_()
		real_labels = tmp.scatter_(dim=1, index=torch.LongTensor(real_labels.view(-1, 1)), value=1)
		tmp = torch.FloatTensor(real_labels.size(0), 10).zero_()
		wrong_labels = tmp.scatter_(dim=1, index=torch.LongTensor(wrong_labels.view(-1, 1)), value=1)
		real_labels = real_labels.to(device)
		wrong_labels = wrong_labels.to(device)

		# 生成数据
		# 用正态分布中采样batch_size个随机噪声
		z = torch.randn([batch_size, z_dim]).to(device)
		# 生成 batch_size 个 ont-hot 标签
		c = torch.FloatTensor(batch_size, 10).zero_()
		c = c.scatter_(dim=1, index=torch.LongTensor(np.random.choice(10, batch_size).reshape([batch_size, 1])), value=1)
		c = c.to(device)
		# 生成数据
		fake_images = generator(z,c)

		# 计算判别器损失，并优化判别器
		real_loss = bce(discriminator(real_images, real_labels), ones)
		fake_loss_1 = bce(discriminator(real_images, wrong_labels), zeros)
		fake_loss_2 = bce(discriminator(fake_images.detach(), c), zeros)
		d_loss = real_loss + fake_loss_1 + fake_loss_2

		d_optimizer.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		# 计算生成器损失，并优化生成器
		g_loss = bce(discriminator(fake_images, c), ones)

		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()

		# 输出损失
		print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

	# 把生成器设置为测试模型，生成效果图并保存
	generator = generator.eval()
	fixed_fake_images = generator(fixed_z, fixed_c)
	save_image(fixed_fake_images, 'improved_cgan_images/{}.png'.format(epoch), nrow=10, normalize=True)

