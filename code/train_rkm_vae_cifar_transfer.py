from utils import create_dirs, convert_to_imshow_format
from stiefel_rkm_model_cifar_pretrained import *
import logging
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Model Settings =================================================================================================
parser = argparse.ArgumentParser(description='St-RKM Model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_name', type=str, default='mnist',
                    help='Dataset name: mnist/fashion-mnist/svhn/dsprites/3dshapes/cars3d/cifar10/imagenette/stl10')
parser.add_argument('--h_dim', type=int, default=10, help='Dim of latent vector')
parser.add_argument('--capacity', type=int, default=64, help='Conv_filters of network')
parser.add_argument('--mb_size', type=int, default=256, help='Mini-batch size')
parser.add_argument('--x_fdim1', type=int, default=256, help='Input x_fdim1')
parser.add_argument('--x_fdim2', type=int, default=50, help='Input x_fdim2')
parser.add_argument('--c_accu', type=float, default=1, help='Input weight on recons_error')
parser.add_argument('--noise_level', type=float, default=1e-3, help='Noise-level')
parser.add_argument('--loss', type=str, default='deterministic', help='loss type: deterministic/noisyU/splitloss')
parser.add_argument('--transfer_learning', dest='use_transfer_learning', action='store_true', help='Use pre-training')
parser.add_argument('--no_transfer_learning', dest='use_transfer_learning', action='store_false', help='Do not use pre-training')

parser.add_argument('--transfer_learning_arch', type=str, default='resnet50', help='pre-trained architecture to use: one of those available in torchvision.models, e.g.) resnet50, resnet152, inception_v3, vgg19, etc.')
parser.add_argument('--use_checkpoint', dest='resume_from_checkpoint', action='store_true', help='Start from checkpoint in case of crash or other early termination, i.e. continue training.')
parser.add_argument('--no_use_checkpoint', dest='resume_from_checkpoint', action='store_false', help='Do not start from checkpoint, i.e. start from scratch.')

parser.set_defaults(use_transfer_learning=False)
parser.set_defaults(resume_from_checkpoint=False)

# Training Settings =============================
parser.add_argument('--lr', type=float, default=2e-4, help='Input learning rate for ADAM optimizer')
parser.add_argument('--lrg', type=float, default=1e-4, help='Input learning rate for Cayley_ADAM optimizer')
parser.add_argument('--max_epochs', type=int, default=1000, help='Input max_epoch')
parser.add_argument('--proc', type=str, default='cuda', help='device type: cuda or cpu')
parser.add_argument('--workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle dataset: True/False')

opt = parser.parse_args()
# ==================================================================================================================
#opt.start_from_checkpoint = False

device = torch.device(opt.proc)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

ct = time.strftime("%Y%m%d-%H%M")
dirs = create_dirs(name=opt.dataset_name, ct=ct)
dirs.create()

# noinspection PyArgumentList
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler('log/{}/{}_{}.log'.format(opt.dataset_name, opt.dataset_name, ct)),
                              logging.StreamHandler()])

""" Load Training Data """
xtrain, ipVec_dim, nChannels = get_dataloader(args=opt)

""" Visualize some training data """
# perm1 = torch.randperm(len(xtrain.dataset))
# it = 0
# fig, ax = plt.subplots(5, 5)
# for i in range(5):
#     for j in range(5):
#         ax[i, j].imshow(convert_to_imshow_format(xtrain.dataset[perm1[it]][0].numpy()))
#         it+=1
# plt.suptitle('Ground Truth Data')
# plt.setp(ax, xticks=[], yticks=[])
# plt.show()

ngpus = torch.cuda.device_count()

#ngpus = 0 # uncomment this value and comment the above to force cpu usage in case of unsupported CUDA version on your GPU - this is sadly the simplest solution
if opt.resume_from_checkpoint:
  list_of_files = glob.glob('./cp/' + opt.dataset_name + '/*') # * means all if need specific format then *.csv
  print('./cp/' + opt.dataset_name + '/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  print('loading from last checkpoint: ' + str(latest_file))

  sd_mdl = torch.load(latest_file)

  rkm = RKM_Stiefel_Transfer(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels, ngpus=ngpus).to(device)
  rkm.load_state_dict(sd_mdl['rkm_state_dict'])
  param_g, param_e1 = param_state(rkm)

  t = sd_mdl['epochs']
  optimizer1 = stiefel_opti(param_g, opt.lrg)
  optimizer2 = torch.optim.Adam(param_e1, lr=opt.lr, weight_decay=0)

  optimizer1.load_state_dict(sd_mdl['optimizer1'])
  optimizer2.load_state_dict(sd_mdl['optimizer2'])
  Loss_stk = sd_mdl['Loss_stk']
  cost = Loss_stk[-1][0]
  l_cost = Loss_stk[-1][0]
else:
  rkm = RKM_Stiefel_Transfer.load_from_checkpoint('../out/cifar10/cifar10_256.ckpt')

  rkm._reset_manifold_param()
  rkm._freeze_decoder_weights()
  rkm._freeze_encoder_weights()
  # Accumulate trainable parameters in 2 groups:
  # 1. Manifold_params 2. Network params
  param_g, param_e1 = param_state(rkm)

  optimizer1 = stiefel_opti(param_g, opt.lrg)
  optimizer2 = torch.optim.Adam(param_e1, lr=opt.lr, weight_decay=0)
  Loss_stk = np.empty(shape=[0, 3])
  cost, l_cost = np.inf, np.inf  # Initialize cost
  t = 1

logging.info(rkm)
logging.info(opt)
logging.info('\nN: {}, mb_size: {}'.format(len(xtrain.dataset), opt.mb_size))
logging.info('We are using {} GPU(s)!'.format(ngpus))

# Train =========================================================================================================
start = datetime.now()
is_best = False

while cost > 1e-10 and t <= opt.max_epochs:  # run epochs until convergence or cut-off
    avg_loss, avg_f1, avg_f2 = 0, 0, 0

    for _, sample_batched in enumerate(tqdm(xtrain, desc="Epoch {}/{}".format(t, opt.max_epochs))):

        loss, f1, f2 = rkm(sample_batched[0].to(device))
        

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        optimizer1.step()

        avg_loss += loss.item()
        avg_f1 += f1.item()
        avg_f2 += f2.item()
    cost = avg_loss

    # Remember lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    dirs.save_checkpoint({
        'epochs': t,
        'rkm_state_dict': rkm.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'Loss_stk': Loss_stk,
    }, is_best)

    logging.info('Epoch {}/{}, Loss: [{}], Kpca: [{}], Recon: [{}]'.format(t, opt.max_epochs, cost, avg_f1, avg_f2))
    print('Epoch {}/{}, Loss: [{}], Kpca: [{}], Recon: [{}]'.format(t, opt.max_epochs, cost, avg_f1, avg_f2))
    Loss_stk = np.append(Loss_stk, [[cost, avg_f1, avg_f2]], axis=0)
    t += 1

logging.info('Finished Training. Lowest cost: {}'
             '\nLoading best checkpoint [{}] & computing sub-space...'.format(l_cost, dirs.dircp))
# ==================================================================================================================

# Load Checkpoint
sd_mdl = torch.load('cp/{}/{}'.format(opt.dataset_name, dirs.dircp))
rkm.load_state_dict(sd_mdl['rkm_state_dict'])

h, U = final_compute(model=rkm, args=opt, ct=ct)
logging.info("\nTraining complete in: " + str(datetime.now() - start))

# Save Model and Tensors ======================================================================================
torch.save({'rkm': rkm,
            'rkm_state_dict': rkm.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'Loss_stk': Loss_stk,
            'opt': opt,
            'h': h, 'U': U}, 'out/{}/{}'.format(opt.dataset_name, dirs.dirout))
logging.info('\nSaved File: {}'.format(dirs.dirout))
