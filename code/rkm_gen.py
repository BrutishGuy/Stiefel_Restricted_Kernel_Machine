import warnings

warnings.filterwarnings("ignore")
import cv2
import time
import utils
import scipy.misc
import torch
import argparse
from dataloader import *
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
from SWD_distance import *
import imageio

# Enter path to load the pre-trained model (see README.md file)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--filename', type=str, default='mnist_Trained_rkm_20200320-0837', help='Enter Filename')
parser.add_argument('--dataset_name', default='mnist', type=str, help='Enter Filename')
opt_gen = parser.parse_args()

sd_mdl = torch.load('out/{}/{}.tar'.format(opt_gen.dataset_name, opt_gen.filename),
                    map_location=lambda storage, loc: storage)

sd_mdl = torch.load('../out/galaxyzoo/galaxyzoorun2_Trained_rkm_20210812-1432.tar', map_location=lambda storage, loc: storage)
opt.filename = 'galaxyzoorun2_Trained_rkm_20210812-1432'
opt.dataset_name='galaxyzoorun2'

sd_mdl = torch.load('../out/cifar10subset/cifar10subset_Trained_rkm_20210811-2221.tar', map_location=lambda storage, loc: storage)
opt.filename = 'cifar10subset_Trained_rkm_20210811-2221'
opt.dataset_name='cifar10subset'

rkm = sd_mdl['rkm']
rkm.load_state_dict(sd_mdl['rkm_state_dict'])
h = sd_mdl['h']
U = sd_mdl['U']
opt = sd_mdl['opt']

""" Load Data """
opt.mb_size = 500
opt.workers = 16
opt.shuffle = True
opt = argparse.Namespace(**vars(opt), **vars(opt_gen))
xtrain, _, nChannels = get_dataloader(args=opt)

WH = next(iter(xtrain))[0].shape[2]  # Number of channels in image

ct = time.strftime("%Y%m%d-%H%M")
def convert_to_imshow_format(image):
    # convert from CHW to HWC
    return image.transpose(1, 2, 0)

# GENERATION ================================================
with torch.no_grad():
    if 'Loss_stk' in sd_mdl:
        Loss_stk = sd_mdl['Loss_stk']
        plt.figure()
        plt.plot(np.log(Loss_stk[:, 0]), 'g-', label='Total Loss')  # log-plot
        plt.plot(np.log(Loss_stk[:, 1]), 'r*-', label='KPCA Recon.')  # log-plot
        plt.plot(np.log(Loss_stk[:, 2]), 'b.-', label='Encoder-decoder')  # log-plot
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Log-Loss')
        plt.title('Loss evolution')

    # # Visualize correlatedness of latent variables
    cov = torch.mm(torch.t(h), h)
    print('Cov(H):\n {}'.format(cov))
    plt.figure()
    plt.imshow(cov.detach().cpu().numpy())
    plt.title('Cov(H)')
    plt.show()

    # # Visualize quality of reconstructed samples
    if opt.dataset_name == 'cifar10' or opt.dataset_name == 'cifar10subset':
        perm1 = torch.randperm(len(xtrain.sampler))
    else:
        perm1 = torch.randperm(len(xtrain.dataset))
    m = 5
    fig2, ax = plt.subplots(m, m)
    it = 0
    for i in range(m):
        for j in range(m):
            ax[i, j].imshow(utils.convert_to_imshow_format(xtrain.dataset[perm1[it]][0].numpy()))
            it += 1
    plt.suptitle('Ground Truth')
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()

    fig1, ax = plt.subplots(m, m)
    x_gen = rkm.decoder(torch.mm(h[perm1[:m * m], :], U.t()).float()).detach().numpy().reshape(-1, nChannels, WH, WH)
    it = 0
    for i in range(m):
        for j in range(m):
            ax[i, j].imshow(utils.convert_to_imshow_format(x_gen[it, :, :, :]))
            it += 1
    plt.suptitle('Reconstructed samples')
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()

    # # Scatter plot of latent variables with histogram ===================
    utils.scatter_w_hist(h)

    # # Interpolation along principal components ================
    for i in range(opt.h_dim):
        dim = i
        m = 35  # Number of steps
        mul_off = 0.0  # (for no-offset, set multiplier to 0)

        # Manually set the linspace range or get from Unit-Gaussian
        lambd = torch.linspace(-5, 5, steps=m)
        # lambd = torch.linspace(*utils._get_traversal_range(0.475), steps=m)

        uvec = torch.FloatTensor(torch.zeros(h.shape[1]))
        uvec[dim] = 1  # unit vector
        yoff = mul_off * torch.ones(h.shape[1]).float()
        yoff[dim] = 0

        yop = yoff.repeat(lambd.size(0), 1) + torch.mm(torch.diag(lambd), uvec.repeat(lambd.size(0), 1))  # Traversal vectors
        x_gen = rkm.decoder(torch.mm(yop, U.t()).float()).detach().numpy().reshape(-1, nChannels, WH, WH)

        # Save Images in the directory
        if not os.path.exists('Traversal_imgs/{}/{}/{}'.format(opt.dataset_name, opt.filename, dim)):
            os.makedirs('Traversal_imgs/{}/{}/{}'.format(opt.dataset_name, opt.filename, dim))

        for j in range(x_gen.shape[0]):
            imageio.imwrite(
                'Traversal_imgs/{}/{}/{}/{}im{}.png'.format(opt.dataset_name, opt.filename, dim, dim, j),
                utils.convert_to_imshow_format(x_gen[j, :, :, :]))

    print('Traversal Images saved in: Traversal_imgs/{}/{}/'.format(opt.dataset_name, opt.filename))

    """ Uncomment the following for random generation and associated metrics """
    #Random samples from dist. over H ============================================================
    utils.gen_gt_imgs(dataset_name=opt.dataset_name, xtrain=xtrain)  # Generate Ground-truth images
    for i in tqdm(range(10)):  # Fit a Gaussian and generate random-images
        if not os.path.exists('Rand_gen_imgs/{}/{}/{}/{}'.format(opt.dataset_name, opt.filename, ct, i)):
            os.makedirs('Rand_gen_imgs/{}/{}/{}/{}'.format(opt.dataset_name, opt.filename, ct, i))
    
        print('Generating random images')
        n_samples = 8000
        gmm = GMM(n_components=1, covariance_type='full').fit(h.detach().cpu().numpy())
        z = torch.FloatTensor(gmm.sample(n_samples)[0])
    
        x_gen = rkm.decoder(torch.mm(z, U.t())).detach().cpu().numpy().reshape(-1, nChannels, WH, WH)
    
        for iter in range(n_samples):
            imageio.imwrite(
                'Rand_gen_imgs/{}/{}/{}/{}/Rand_samp_im{}_{}.png'
                    .format(opt.dataset_name, opt.filename, ct, i, iter, time.strftime("%Y%m%d-%H%M")),
                utils.convert_to_imshow_format(x_gen[iter, :, :, :]))

    indx1 = 245
    indx2 = 45
    indx3 = 3
    indx4 = 56
    
    y1 = h[indx1, :]
    y2 = h[indx2, :]
    y3 = h[indx3, :]
    y4 = h[indx4, :]
    
    # # 2-D Interpolation%%%%%%%%%%%%
    m = 10
    T, S = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    T = np.ravel(T, order="F")
    S = np.ravel(S, order="F")
    
    fig5, ax = plt.subplots(m, m)
    it = 0
    for i in range(m):
        for j in range(m):
    
            # weights
            lambd = np.flip(np.hstack((S[it]*T[it], (1-S[it])*T[it], S[it]*(1-T[it]), (1-S[it])*(1-T[it]))), 0)
            # computation
            yop = lambd[0] * y1 + lambd[1] * y2 + lambd[2] * y3 + lambd[3] * y4
            yop = torch.FloatTensor(yop)
            x_gen = rkm.decoder(torch.mv(U, yop)).detach().cpu().numpy().reshape( nChannels, WH, WH)
    
            ax[i, j].imshow(convert_to_imshow_format(x_gen))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            it += 1
    plt.suptitle('Interpolation in latent space')
    plt.show()

    # # SWD Computation ====================
    # im_vae1 = []
    # dist = []
    # gt = 'gt_imgs/{}'.format(opt.dataset_name)
    # gen = 'Rand_gen_imgs/{}/{}/{}'.format(opt.dataset_name, opt.filename, ct)
    #
    # files = glob.glob("{}/*.png".format(gt))  # your image path
    # for myFile in files:
    #     image = cv2.imread(myFile)
    #     im_vae1.append(image)
    #
    # im_vae1 = np.array(im_vae1, dtype='float32')
    # images1 = im_vae1.astype('float32') / 255
    #
    # for i in range(10):
    #     im_vae2 = []
    #     files2 = glob.glob("{}/{}/*.png".format(gen, i))  # your image path
    #     for myFile in files2:
    #         image = cv2.imread(myFile)
    #         im_vae2.append(image)
    #     im_vae2 = np.array(im_vae2, dtype='float32')
    #     images2 = im_vae2.astype('float32') / 255
    #
    #     dist.append(sliced_wasserstein_distance(generated_images=images2, x=images1, sample_nr=8000))
    #     print('Dist: {}'.format(dist))
    # print('Mean: {}, Std: {}'.format(np.mean(dist), np.std(dist)))
    #
    # # # FID Computation ======================
    # fid_compu = utils.fid(gt='gt_imgs/{}'.format(opt.dataset_name),
    #                       gen='Rand_gen_imgs/{}/{}/{}'.format(opt.dataset_name, opt.filename, ct))
    # fid_compu.comp_fid()
