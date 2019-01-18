#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os


##########################################################
# 超参数
##########################################################
gpu_id = None #‘0’
if gpu_id is not None:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
if os.path.exists('gan_images') is False:
	os.makedirs('gan_images')
z_dim = 100
batch_size = 64
learning_rate = 0.0002
total_epochs = 200


##########################################################
# 定义模型
##########################################################
class Discriminator(nn.Module):
	'''全连接判别器，用于1x28x28的MNIST数据'''
	def __init__(self):
		super(Discriminator, self).__init__()

		layers = []
		# 第一层
		layers.append(nn.Linear(in_features=28*28, out_features=512, bias=True))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 第二层
		layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
		layers.append(nn.LeakyReLU(0.2, inplace=True))
		# 输出层
		layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
		layers.append(nn.Sigmoid())

		self.model = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		validity = self.model(x)
		return validity

class Generator(nn.Module):
	'''全连接生成器，用于1x28x28的MNIST数据'''
	def __init__(self, z_dim):
		super(Generator, self).__init__()

		layers= []
		# 第一层
		layers.append(nn.Linear(in_features=z_dim, out_features=128))
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
		layers.append(nn.Tanh()) #[-1,1]

		self.model = nn.Sequential(*layers)

	def forward(self, z):
		x = self.model(z)
		x = x.view(-1, 1, 28, 28)
		return x

# 初始化构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)


##########################################################
# 准备工作
##########################################################
# 初始化二值交叉熵损失
bce = nn.BCELoss().to(device)
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.999])
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999])

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 随机产生100个向量，用于生成效果图
fixed_z = torch.randn([100, z_dim]).to(device)


##########################################################
# 开始训练，一共训练total_epochs
##########################################################
for epoch in range(total_epochs):

	# 在训练阶段，把生成器设置为训练模式；对应于后面的，在测试阶段，把生成器设置为测试模式
	generator = generator.train()

	# 训练一个epoch
	for i, data in enumerate(dataloader):

		# 加载真实数据，不加载标签
		real_images, _ = data
		real_images = real_images.to(device)

		# 从正态分布中采样batch_size个噪声，然后生成对应的图片
		z = torch.randn([batch_size, z_dim]).to(device)
		fake_images = generator(z)

		# 计算判别器损失，并优化判别器
		real_loss = bce(discriminator(real_images), ones)
		fake_loss = bce(discriminator(fake_images.detach()), zeros)
		d_loss = real_loss + fake_loss

		d_optimizer.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		# 计算生成器损失，并优化生成器
		g_loss = bce(discriminator(fake_images), ones)

		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()

		# 输出损失
		print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

	# 把生成器设置为测试模型，生成效果图并保存
	generator = generator.eval()
	fixed_fake_images = generator(fixed_z)
	save_image(fixed_fake_images, 'gan_images/{}.png'.format(epoch), nrow=10, normalize=True)

