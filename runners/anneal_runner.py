import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import math
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from Datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image

__all__ = ['AnnealRunner']


class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            #dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
            #                 transform=tran_transform)
            dataset = CIFAR10("../../CIFAR10/data",train = True,transform=tran_transform)
            #test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
            #                       transform=test_transform)
            test_dataset = CIFAR10("../../CIFAR10/data",train = False,transform=tran_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score,device_ids = [1, 2, 3, 4])

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256. # Adding random noise  to the data
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images


    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

        
    def calculate_gamma(self, t, gamma_max):
        # The following schedule was suggested by Sutskever et. al., 2013
        # γmax was chosen from {0.999, 0.995, 0.99, 0.9, 0}
        base = 1 / (1-0.1)
        term = 1 - base ** (-1 - math.log2(math.floor(t/20) + 1))
        gamma_t = min(term, gamma_max)
        return gamma_t
    

    def anneal_Langevin_dynamics_with_momentum(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002, gamma=0.0000):
        images = []
        V = torch.zeros_like(x_mod)
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    
                    ## Momentum update
                    gamma = self.calculate_gamma(c*n_steps_each, 0.99)
                    V = gamma*V + step_size * grad

                    x_mod = x_mod + V + noise


                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images



    def anneal_Langevin_dynamics_with_nesterov(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002,gamma = 0.0):
        images = []
        V = torch.zeros_like(x_mod)
        gamma = gamma
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                #print(labels[:5])
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                # if c<2:
                #     gamma = 0.05
                # elif c<3:
                #     gamma = 0.07
                # elif c<5:
                #     gamma = 0.1
                # elif c<6:
                #     gamma = 0.12
                # elif c<7:
                #     gamma = 0.15
                # elif c<8:
                #     gamma = 0.17
                # elif c<9:
                #     gamma = 0.18
                # else:
                #     gamma = 0.2
                # # elif c<7:
                # #     gamma = 0.32
                # # elif c<8:
                # #     gamma = 0.7
                # # else:
                # #     gamma = 0.8
                # print("------C == {} ------ Gamma == {}".format(c,gamma))
                # gamma+=0.02
                if c>5:
                    gamma+=0.02
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    # grad = scorenet(x_mod, labels)

                    
                    # gamma = self.calculate_gamma(c*n_steps_each, 0.5)
                    # gamma = 0.15
                    grad = scorenet(x_mod+gamma*V,labels)
                    V = gamma*V + step_size * grad
                    x_mod = x_mod + V + noise

                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images


    def find_fid(self):

        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle
        import shutil

        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score,device_ids = [0])

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        fids = {}

        if self.args.T==10:
            gammas = [0.1, 0.15, 0.23, 0.28]
            # gammas = [0.0]
        elif self.args.T==100:
            gammas = [0.0]
        else:
            #gammas = [0.07,0.1,0.13,0.16,0.2,0.23,0.28]
            gammas = [0.054]
            #gammas = [0.02,0.03,0.05]

        num_samples = 10000
        batch_size = 128

        if self.config.data.dataset == 'MNIST':
            channels = 1
            image_size = 28
        else:
            channels = 3
            image_size = 32

        for gamma in tqdm.tqdm(gammas,
                              desc="processing gammas"):
            num_iters = num_samples // batch_size
            output_path = os.path.join(self.args.image_folder, 'T_{}gamma_{}'.format(self.args.T,gamma))
            os.makedirs(output_path, exist_ok=True)

            for i in range(num_iters):
                init_samples = torch.rand(batch_size, channels,
                                          image_size, image_size,
                                          device=self.config.device)

                all_samples = self.anneal_Langevin_dynamics_with_nesterov(init_samples, score, sigmas, self.args.T, 0.00002,gamma)
                

                print("--------------- Sampled Images  -------------")
                print("\n------------- Saving images ---------------\n")
                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(channels,
                                         image_size,
                                         image_size)
                    
                    if self.config.data.logit_transform:
                        sample = torch.sigmoid(sample)

                    

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(i*batch_size + id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[gamma] = fid
            print("------- Number of sampled images == {} -----".format(i*batch_size + id))
            print("Gamma: {}, fid: {}".format(gamma, fid))
            # Open a file for appending
            file_path = "fid_results.txt"
            with open(file_path, "a") as file:
                #file.write("10000, 128, gamma of {} and incrementing by 0.02 where number of iterations for each noise level is {}, FID comes out to be {}\n".format(gamma, self.args.T,fid))
                file.write("10000, 128, fid with 500 iterations and no gamma is {}\n".format(fid))
            shutil.rmtree(output_path)

        with open(os.path.join(self.args.image_folder, 'T_{}fids.pickle'.format(self.args.T)), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def calculate_fid(self):
        from evaluation.fid_score import get_fid, get_fid_stats_path       

        output_path = '/home/nishit/ncsn_ICASP/fid_sample/cifar10/nesterov/T_10gamma_0.1'
        stat_path = get_fid_stats_path(self.args, self.config, download=True)
        fid = get_fid(stat_path, output_path)
        print(fid)
        file_path = "fid_results.txt"
        with open(file_path, "a") as file:
            file.write("hello, writting from calculate_fid function")

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score,device_ids = [0])

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        grid_size = 10

        imgs = []
        if self.config.data.dataset == 'MNIST':
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)

            # original n_steps_each = 100, step_lr = 0.00002
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 10, 0.00002,gamma=0.0)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                #save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                #torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))


        else:
            samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

            # original n_steps_each = 100, step_lr = 0.00002
            all_samples = self.anneal_Langevin_dynamics_with_nesterov(samples, score, sigmas, self.args.T, 0.00002,0.05)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                #save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
                #torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        # original duration =1
        path = self.args.image_folder+'/'+str(self.args.T)+'_step_gamma{0.05,+0.02}'
        #path = self.args.image_folder
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(image_grid, os.path.join(path, 'image_{}.png'.format(i)), nrow=10)
        imgs[0].save(os.path.join(path, "movie.gif"), save_all=True, append_images=imgs[1:], duration=800, loop=0)

    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []
        original = refer_image.clone()
        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32 ,32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    img = torch.zeros(5,7,3,32,32)
                    img[:,0,:,:,:16] = original[:,:,:,:16]
                    img[:,1:6,:,:,:] = x_mod.view(5,5,3,32,32)
                    img[:,6,:,:,:] = original
                    if c==0 and s==0:
                        image_grid = make_grid(img.view(-1,3,32,32), nrow=7)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(c))) 
                    images.append(torch.clamp(img.view(-1,3,32,32), 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score,device_ids = [1, 2, 3, 4])

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        imgs = []
        if self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                #dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                #                  transform=transform)
                dataset = CIFAR10("../../CIFAR10/data",train = True,transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            _, _ = next(data_iter)
            refer_image,_ = next(data_iter)

            
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(5, 5, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(35, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=7)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)
 
                #save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                #torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))


        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
