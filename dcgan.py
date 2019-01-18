#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os


# 超参数
gpu_id = None
if gpu_id is not None:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
if os.path.exists('dcgan_images') is False:
	os.makedirs('dcgan_images')
z_dim = 100
batch_size = 64
learning_rate = 0.0002
total_epochs = 100


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
	'''滑动卷积判别器'''
	def __init__(self):
		super(Discriminator, self).__init__()

		# 定义卷积层
		conv = []
		# 第一个滑动卷积层，不使用BN，LRelu激活函数
		conv.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1))
		conv.append(nn.LeakyReLU(0.2, inplace=True))
		# 第二个滑动卷积层，包含BN，LRelu激活函数
		conv.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1))
		conv.append(nn.BatchNorm2d(32))
		conv.append(nn.LeakyReLU(0.2, inplace=True))
		# 第三个滑动卷积层，包含BN，LRelu激活函数
		conv.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
		conv.append(nn.BatchNorm2d(64))
		conv.append(nn.LeakyReLU(0.2, inplace=True))
		# 第四个滑动卷积层，包含BN，LRelu激活函数
		conv.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1))
		conv.append(nn.BatchNorm2d(128))
		conv.append(nn.LeakyReLU(0.2, inplace=True))
		# 卷积层
		self.conv = nn.Sequential(*conv)

		# 全连接层+Sigmoid激活函数
		self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=1), nn.Sigmoid())

	def forward(self, x):
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		validity = self.linear(x)
		return validity

class Generator(nn.Module):
	'''反滑动卷积生成器'''

	def __init__(self, z_dim):
		super(Generator, self).__init__()

		self.z_dim = z_dim
		layers = []

		# 第一层：把输入线性变换成256x4x4的矩阵，并在这个基础上做反卷机操作
		self.linear = nn.Linear(self.z_dim, 4*4*256)
		# 第二层：bn+relu
		layers.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0))
		layers.append(nn.BatchNorm2d(128))
		layers.append(nn.ReLU(inplace=True))
		# 第三层：bn+relu
		layers.append(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1))
		layers.append(nn.BatchNorm2d(64))
		layers.append(nn.ReLU(inplace=True))
		# 第四层:不使用BN，使用tanh激活函数
		layers.append(nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=2))
		layers.append(nn.Tanh())

		self.model = nn.Sequential(*layers)

	def forward(self, z):
		# 把随机噪声经过线性变换，resize成256x4x4的大小
		x = self.linear(z)
		x = x.view([x.size(0), 256, 4, 4])
		# 生成图片
		x = self.model(x)
		return x

# 构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)
# 使用均值为0，方差为0.02的正态分布初始化神经网络
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 初始化二值交叉熵损失
bce = torch.nn.BCELoss().to(device)
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.999])
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999])

# 加载MNIST数据集
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 随机产生100个向量，用于生成效果图
fixed_z = torch.randn([100, z_dim]).to(device)

# 开始训练，一共训练total_epochs
for epoch in range(total_epochs):

	# 在训练阶段，把生成器设置为训练模型；对应于后面的，在测试阶段，把生成器设置为测试模型
	generator = generator.train()

	# 训练一个epoch
	for i, data in enumerate(dataloader):

		# 加载真实数据，不加载标签
		real_images, _ = data
		real_images = real_images.to(device)

		# 用正态分布中采样batch_size个噪声，然后生成对应的图片
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
		print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

	# 把生成器设置为测试模型，生成效果图并保存
	generator = generator.eval()
	fixed_fake_images = generator(fixed_z)
	save_image(fixed_fake_images, 'dcgan_images/{}.png'.format(epoch), nrow=10, normalize=True)

