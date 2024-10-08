import os
import time
from dataset import  Generator
from utils import PSNR, progress_bar, clock, rgb2y_uint8
from matlab import imresize, convertDouble2Byte as d2int
import numpy as np
from skimage.io import imread, imsave
from skimage.util import img_as_float64, img_as_float32, img_as_ubyte
import torch
import torch.nn as nn
import torch.optim as optim
from ASFSR.ASFSR import Net
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from skimage.color import rgb2ycbcr


def val_psnr(model, val_path, scale, th, dilker, dilation, gray=True):
    images = sorted(os.listdir(val_path))
    avg_psnr = 0
    for image in images:

        img = imread(val_path + image)
        try:
            h, w, c = img.shape
            img = rgb2ycbcr(img)[:, :, 0:1]
        except:
            h, w = img.shape
            img = np.expand_dims(img, axis=-1)

        img = np.float64(img)/255
        h, w, c = img.shape


        orig = img[0:h - (h % scale), 0:w - (w % scale)]
        lr = imresize(orig, scalar_scale=1 / scale, method='bicubic')
        #bic = imresize(lr, scalar_scale=scale, method='bicubic')

        lr_data= np.moveaxis(lr, -1, 0)

        lr_data = np.expand_dims(lr_data, axis=0)
        lr_data= torch.from_numpy(lr_data).float().to(device)

        sr = model(lr_data, th, dilker, dilation)
        sr = sr[0].cuda().data.cpu().numpy()

        if gray == True: sr = sr[0]
        else: sr = np.moveaxis(sr, 0, -1)

        orig = d2int(orig)
        sr = d2int(sr)

        if gray == True:
            gray_orig = orig[:,:,0]
            gray_sr = sr
        else:
            gray_orig = rgb2y_uint8(orig)
            gray_sr = rgb2y_uint8(sr)

        avg_psnr += PSNR(gray_orig, gray_sr, boundary=scale, max_value=255)
        # if image == 'butterfly.bmp' and scale == 2:
        #     try:os.mkdir('butterfly')
        #     except:pass
        #     imsave('butterfly/GT.bmp', (orig))
        #     imsave('butterfly/sr.bmp', (sr))

    return avg_psnr/len(images)

#===============
train_path = 'dataset/LR_HR/scale_x2/dataset_cpy1.h5'
val_path = 'C:/Users/user/Desktop/sr/Set5/'

save_as = 'ASFSR/checkpoint/scale_x2/th_rand_dilk3.pth'

dilation= True
dilker = 3
r = 4

th_begin = 0.01
th_end = 0.10
range_gap = 0.01
th_n = int((th_end - th_begin)/range_gap)+1

ths = []
for i in range(th_n):
    ths.append(th_begin+(range_gap*i))

device = torch.device('cuda')
#===============


train_scale = 2
test_scale = 2

batch_size = 32
itr_parts = 80#400
n_steps = 10000

initial_lr = 7e-4#1e-3    nvbnvnvn


model = Net(scale=train_scale, r= r)


model = model.float()
model.to(device)

#summary(model,(3, 64, 64))

mae = nn.MSELoss().to(device)

lr = initial_lr

optimizer = optim.Adam(model.parameters(), lr= initial_lr)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = (n_steps * itr_parts), eta_min=1e-10)

print('Number of Parameters:', sum(p.numel() for p in model.parameters()))

lr_hr_data = Generator(device = device, file_path=train_path, scale=2, num_workers=0, batch_size=batch_size, shuffle=True)


start_from = 1
best = 0

for itr_part in range(start_from, itr_parts+1):
    print('itr_Part {}/{}'.format(itr_part, itr_parts))

    model.train()
    avg_loss = 0
    strt_time = time.time()

    for step in range (1, n_steps+1):
        optimizer.zero_grad()

        lr_batch, hr_batch = next(lr_hr_data)
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
        lr_batch, hr_batch = lr_batch.float() / 255, hr_batch.float() / 255

        th = random.choice(ths)
        predict = model(lr_batch, th, dilker, dilation, eval=False)

        loss = mae(predict, hr_batch)

        loss_limit = 10
        if float(loss) < loss_limit and float(loss) > -loss_limit:
            loss.backward()
            optimizer.step()

        #=======learning rate update=============
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        # =======================================

        avg_loss += float(loss)
        progress_bar(step, n_steps, float(loss))

    progress_bar(n_steps, n_steps, float(avg_loss / n_steps))

    with torch.no_grad():
        model.eval()
        tmp =0
        for th in ths:
            avg_val_psnr = val_psnr(model, val_path, scale=test_scale, th=th, dilker=dilker, dilation=dilation)
            print('-'+str(th) + '_val_psnr: {:.5f}'.format(avg_val_psnr), end=' ')
            tmp += avg_val_psnr

        avg_val_psnr = tmp/len(ths)
        if avg_val_psnr > best:
            torch.save(model.state_dict(), save_as)
            best = avg_val_psnr

    print('- lr: {:.7f}'.format(float(lr)), end=' ')

    stp_time = time.time()
    min, sec = clock(int(stp_time - strt_time))
    print('- [{}:{}s]'.format(min, sec))


print()
print('Training Finished')
print('checkpoints saved as:', save_as)