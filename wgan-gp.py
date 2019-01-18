#coding:utf-8
import torch
import torch.nn as nn
import torch.autograd as autograd
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
if os.path.exists('wgangp_images') is False:
	os.makedirs('wgangp_images')
z_dim = 100
batch_size = 64
total_epochs = 200
learning_rate = 0.0001
weight_gp = 10
n_critic = 5


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

		# 输出层：相比gan，不需要sigmoid
		layers.append(nn.Linear(in_features=256, out_features=1, bias=True))

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
		layers.append(nn.Tanh())

		self.model = nn.Sequential(*layers)

	def forward(self, z):
		x = self.model(z)
		x = x.view(-1, 1, 28, 28)
		return x

# 初始化构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)

# 计算梯度惩罚正则项
def compute_gradient_penalty(D, real_samples, fake_samples):
	# 在真实样本所在空间 和 生成样本空间 之间采样样本 （通过插值进行采样）
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	# 计算判别器对于这些样本的梯度
    d_interpolates = D(interpolates)
    fake = autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # 计算梯度损失
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 初始化优化器，使用Adam优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.9])
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.9])

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 随机产生100个向量，用于生成效果图
fixed_z = torch.randn([100, z_dim]).to(device)

# 开始训练，一共训练total_epochs
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
		d_gan = - torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_images))
		d_gp = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
		d_loss = d_gan + weight_gp * d_gp
		d_optimizer.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		# 每优化n_critic次判别器， 优化一次生成器
		if i % n_critic == 0:
			fake_images = generator(z)
			g_loss = - torch.mean(discriminator(fake_images))
			g_optimizer.zero_grad()
			g_loss.backward()
			g_optimizer.step()

			# 输出损失
			print ("[Epoch %d/%d] [Batch %d /%d] [D loss: %f %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_gan.item(), d_gp.item(), g_loss.item()))

	# 每训练一个epoch，把生成器设置为测试模型，生成效果图并保存
	generator = generator.eval()
	fixed_fake_images = generator(fixed_z)
	save_image(fixed_fake_images, 'wgangp_images/{}.png'.format(epoch), nrow=10, normalize=True)

