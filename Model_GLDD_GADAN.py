import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class DeformableAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(DeformableAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_res_blocks=5):
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(in_channels, 64, 7, padding=3)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.attention = DeformableAttentionBlock(64)
        self.final = nn.Conv2d(64, out_channels, 7, padding=3)

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.res_blocks(x)
        x = self.attention(x)
        x = torch.tanh(self.final(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.global_fc = nn.Linear(256 * 8 * 8, 1)
        self.local_fc = nn.Linear(256 * 16 * 16, 1)

    def forward(self, x):
        feat = F.leaky_relu(self.conv1(x), 0.2)
        feat = F.leaky_relu(self.conv2(feat), 0.2)
        feat = F.leaky_relu(self.conv3(feat), 0.2)
        B, C, H, W = feat.size()
        global_out = self.global_fc(feat.view(B, -1))
        local_out = self.local_fc(F.interpolate(feat, size=(16, 16)).view(B, -1))
        return global_out, local_out


def Model_GLDD_GADAN(Train_Data, Train_Tar, Test_Data, Test_Tar, epochs=5, batch_size=2):
    # Convert data to torch tensors
    Train_X = torch.tensor(Train_Data, dtype=torch.float32)
    Test_X = torch.tensor(Test_Data, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Tar, dtype=torch.float32)

    # Initialize networks
    G = Generator()
    D = Discriminator()

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    criterion = nn.BCELoss()  # Binary cross-entropy for discriminator

    for epoch in range(epochs):
        for i in range(0, Train_X.shape[0], batch_size):
            real_imgs = Train_X[i:i + batch_size]
            batch_size_curr = real_imgs.size(0)
            valid = torch.ones(batch_size_curr, 1)
            fake = torch.zeros(batch_size_curr, 1)
            # Train Generator
            optimizer_G.zero_grad()
            gen_imgs = G(real_imgs)
            g_loss_global, g_loss_local = D(gen_imgs)
            g_loss = criterion(torch.sigmoid(g_loss_global), valid) + criterion(torch.sigmoid(g_loss_local), valid)
            g_loss.backward()
            optimizer_G.step()
            # Train Discriminator
            optimizer_D.zero_grad()
            real_global, real_local = D(real_imgs)
            fake_global, fake_local = D(gen_imgs.detach())
            d_loss = criterion(torch.sigmoid(real_global), valid) + criterion(torch.sigmoid(real_local), valid) + \
                     criterion(torch.sigmoid(fake_global), fake) + criterion(torch.sigmoid(fake_local), fake)
            d_loss.backward()
            optimizer_D.step()
    # Prediction
    G.eval()
    with torch.no_grad():
        pred = G(Test_X).numpy()
    # Convert predictions to binary
    min_val, max_val = pred.min(), pred.max()
    avg = (min_val + max_val) / 2
    pred_bin = np.zeros_like(pred)
    pred_bin[pred >= avg] = 1
    pred_bin[pred < avg] = 0
    # Evaluate
    Eval = ClassificationEvaluation(Test_Tar, pred_bin)
    return np.asarray(Eval).ravel()
