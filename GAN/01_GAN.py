# MNIST 이미지 생성하는 문제

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
transforms.Normalize(mean=(0.5,), std=(0.5,)) # 픽셀값: 0~1 => -1~1
])

# set dataset and dataloader
mnist = datasets.MNIST(root='./data',
                       download=True,
                       transform=transform)

dataloader = DataLoader(mnist, batch_size=100, shuffle=True)

# 생성자(Generator):
# Random vector Z을 입력으로 받아, 가짜 이미지를 생성하는 함수
# Z: uniform dist or normal dist에서 무작위로 추출된 값
# 단순한 분포를 복잡한 분포로 매핑하는 함수가 생성
# 잠재 공간(Latent space): 충분히 크다면 임의로 생성해도 됨.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28), # MNIST output size
            nn.Tanh()
        )

    # 임의의 MNIST 가짜 이미지를 출력
    def forward(self, x):
        return self.net(x).view(-1, 1, 28, 28) # batch, ch, h, w


# 구분자(Discriminator):
# 이미지를 입력 받고, 그 이미지가 진짜일 확률을 0~1 사이의 숫자로 출력하는 함수
# Activation function: Sigmoid
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid() # 이미지가 진짜일 확률
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.net(x)

if __name__ == "__main__":
    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss() # 진짜 이미지인지, 아닌지 Binary classification

    # 생성자와 구분자의 optimizer
    G_opt = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_opt = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(100):
        avg_D_loss = []
        avg_G_loss = []

        for step, (real_data, _) in enumerate(dataloader):
            batch_size = real_data.size(0)

            # real_data = real_data.to(device)
            real_data = Variable(real_data)


            ########### 구분자 학습 ###########
            # 구분자: 진짜 이미지 입력 -> 1에 가까운 확률값
            #         가짜 이미지 입력 -> 0에 가까운 확률값
            # loss = (진짜 이미지 입력 시 출력값과 1의 차이) + (가짜 이미지 입력 시 출력값과 0의 차이)
            # 이 loss를 최소화 하는 방향

            target_real = Variable(torch.ones(batch_size, 1)) # 진짜일 때 정답지
            target_fake = Variable(torch.zeros(batch_size, 1))# 가짜일 때 정답지

            # 진짜 이미지가 Discriminator에 들어갔을 때의 result와 loss
            D_result_from_real = D(real_data)
            D_loss_real = criterion(D_result_from_real, target_real)

            # 생성자에게 입력으로 줄 Random vector Z
            # z = torch.randn(batch_size, 100).to(device)
            z = Variable(torch.randn(batch_size, 100))

            # 생성자로 가짜 이미지 생성
            fake_data = G(z)

            # 가짜 이미지가 Discriminator에 들어갔을 때의 result와 loss
            D_result_from_fake = D(fake_data)
            D_loss_fake = criterion(D_result_from_fake, target_fake)

            # loss + forward + backward
            D_loss = D_loss_real + D_loss_fake
            D.zero_grad()
            D_loss.backward()
            D_opt.step()

            ########### 생성자 학습 ###########

            # 생성자에 입력으로 줄 랜덤 벡터 z 생성
            z = Variable(torch.randn(batch_size, 100))

            # 생성자로 가짜 이미지를 생성
            fake_data = G(z)

            # 생성자가 만든 가짜 이미지를 구분자에 입력
            D_result_from_fake = D(fake_data)

            # 생성자의 입장에서 구분자의 출력값이 1에서 멀수록 loss가 높아짐

            # loss + forward + backward
            G_loss = criterion(D_result_from_fake, target_real)
            G.zero_grad()
            G_loss.backward()
            G_opt.step()

            print("[%d/%d] [%d/%d] D loss:%.3f, G loss:%.3f" % (
                epoch+1, 100, step, len(dataloader), D_loss.item(), G_loss.item()
            ))

            avg_D_loss.append(D_loss.item())
            avg_G_loss.append(G_loss.item())

        print("[%d/%d] Avg D loss:%.3f, Avg G loss:%.3f" % (
            epoch+1, 100, np.mean(avg_D_loss), np.mean(avg_G_loss)
        ))





