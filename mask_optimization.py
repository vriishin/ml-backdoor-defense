from models.selector import *
from utils.util import *
from data_loader import *
from config import get_arguments

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(opt, model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=opt.anp_eps)


def train_step(opt, data_loader, model, mask_optimizer, noise_optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    total_correct = 0
    total_loss = 0.0
    for idx, (img, target) in enumerate(data_loader):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        # step 1: calculate the adversarial perturbation for neurons
        if opt.anp_eps > 0.0:
            reset(opt, model, rand_init=True)
            for _ in range(opt.anp_steps):
                noise_optimizer.zero_grad()

                include_noise(model)
                output_noise = model(img)
                loss_noise = - criterion(output_noise, target)

                loss_noise.backward()
                sign_grad(model)
                noise_optimizer.step()

        # step 2: calculate loss and update the mask values
        mask_optimizer.zero_grad()
        if opt.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(img)
            loss_rob = criterion(output_noise, target)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output = model(img)
        loss_nat = criterion(output, target)
        loss = opt.anp_alpha * loss_nat + (1 - opt.anp_alpha) * loss_rob
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
        loss.backward()
        mask_optimizer.step()
        clip_mask(model)

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'Loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(data_loader), losses=losses, top1=top1, top5=top5))

def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]
    return acc_clean, acc_bd

def train(opt):
    # Load models
    print('----------- Network Initialization --------------')

    model_directory = (opt.finetuning_checkpoint_root if opt.with_finetuned_model else opt.tuning_checkpoint_root) + "/"
    model_directory_files = []
    for filename in os.listdir(model_directory):
      file_path = os.path.join(model_directory, filename)
      if not os.path.isdir(file_path):
        model_directory_files.append((file_path, os.path.getctime(file_path)))
    model_directory_files = sorted(model_directory_files, key=lambda x: x[1])
    model_path = model_directory_files[-1][0]
    model, checkpoint_epoch = select_model(dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=model_path,
                                n_classes=opt.num_class,
                                norm_layer=opt.norm_layer)
    model.to(opt.device)
    print('Finish loading model...')

    # Initialize mask optimizer
    parameters = list(model.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    #Need to ensure this list isn't empty to make sure masking information is available in the model
    mask_optimizer = torch.optim.SGD(mask_params, lr=opt.lr_mask_opt, momentum=0.1)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=opt.anp_eps / opt.anp_steps)


    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    if opt.mask_opt_data_type == "clean":

      tf_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          Cutout(1, 3)
      ])


      #The data you use when you want to use when you want to perform mask optimzation using a 'known' clean dataset, rather than the data from the backdoor isolation stage.
      mask_data = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
      mask_data_tf = Dataset_npy(full_dataset=mask_data, transform=tf_train)
      random_sampler = RandomSampler(data_source=mask_data_tf, replacement=True, num_samples=opt.mask_opt_sample_size)
      mask_data_loader = DataLoader(dataset=mask_data_tf,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                sampler = random_sampler
                                                )
    else:
      tf_train = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          Cutout(1, 3)
      ])
      if opt.mask_opt_data_type == "random":
        #The data you use when you want to use when you want to perform mask optimzation using a random sample from the presumed clean data set from the backdoor isolation step.
        data_path_mask = os.path.join(opt.isolate_data_root, "{}_other{}%_examples.npy".format(opt.model_name, 
                                                                100 - opt.isolation_ratio * 100 - opt.mask_isolation_ratio * 100))
        mask_data = np.load(data_path_mask, allow_pickle=True)
        mask_data_tf = Dataset_npy(full_dataset=mask_data, transform=tf_train)
        random_sampler = RandomSampler(data_source=mask_data_tf, replacement=True, num_samples=opt.mask_opt_sample_size)
        mask_data_loader = DataLoader(dataset=mask_data_tf,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  sampler = random_sampler
                                                  )
      else:
        #The data you use when you want to use when you want to perform mask optimzation using the high loss data from the backdoor isolation step.

        data_path_mask = os.path.join(opt.isolate_data_root, "{}_mask_isolation{}%_examples.npy".format(opt.model_name, opt.mask_isolation_ratio * 100))
        mask_data = np.load(data_path_mask, allow_pickle=True)
        mask_data_tf = Dataset_npy(full_dataset=mask_data, transform=tf_train)
        mask_data_loader = DataLoader(dataset=mask_data_tf,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  )

    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    test_process = []
    cur_max = float('-inf')
    for epoch in range(0, opt.mask_opt_epochs):
        # train every epoch
        
        if epoch == 0:
            #before training test firstly
            test(opt, test_clean_loader, test_bad_loader, model, criterion, epoch + 1)

        train_step(opt, mask_data_loader, model, mask_optimizer, noise_optimizer, criterion, epoch + 1)

        # evaluate on testing set
        print('testing the model with optimized mask......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model, criterion, epoch + 1)

        if epoch % opt.mask_opt_interval == 0:
          print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
          print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bad[0], acc_bad[2]))
          log_root = opt.log_root + '/mask_optimization_results_tuning_epochs.csv'
          test_process.append(
            (epoch, acc_clean[0], acc_bad[0], acc_clean[2], acc_bad[2]))
          df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                                 "Test_clean_loss", "Test_bad_loss"))
          df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

        loss_diff = acc_bad[2] - acc_clean[2]
        cur_max = max(cur_max, loss_diff)
        is_best = (cur_max == loss_diff)
        if opt.save:
            # save checkpoint at interval epoch
            if epoch % opt.mask_opt_interval == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'clean_acc': acc_clean[0],
                    'bad_acc': acc_bad[0],
                    'optimizer': optimizer.state_dict(),
                }, epoch, is_best, opt)

    save_mask_scores(model.state_dict(), opt)
    if opt.load_fixed_data:
        tf_compose = transforms.Compose([
            transforms.ToTensor()
        ])
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc.
        poisoned_data = np.load(opt.poisoned_data_path, allow_pickle=True)
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            )
    else:
        poisoned_data, poisoned_data_loader = get_backdoor_loader(opt)

    return poisoned_data, model


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        os.makedirs(opt.mask_optimization_checkpoint_root, exist_ok=True)
        filepath = os.path.join(opt.mask_optimization_checkpoint_root, opt.model_name + r'_mask_optimization_epochs{}.tar'.format(epoch))
        torch.save(state, filepath)
    print('[info] Finish saving the model')


def save_mask_scores(state_dict, opt):
    os.makedirs(opt.mask_score_root, exist_ok=True)
    file_name = os.path.join(opt.mask_score_root, 'mask_values.txt')
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


def main():
    print('----------- Train backdoored model for mask optimization -----------')
    opt = get_arguments().parse_args()
    train(opt)

if (__name__ == '__main__'):
    main()