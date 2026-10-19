from models.selector import *
from utils.util import *
from data_loader import *
from config import get_arguments

def compute_loss_value(opt, poisoned_data, model_ascent):
    # Calculate loss value per example
    # Define loss function
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
            # print(loss.item())

        losses_record.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record))   # get the index of examples by loss value in ascending order

    # Show the lowest 10 loss values
    losses_record_arr = np.array(losses_record)
    print('Top ten loss value:', losses_record_arr[losses_idx[:10]])

    return losses_idx

def isolate_data(opt, poisoned_data, losses_idx):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0
    isolation_ratio = opt.isolation_ratio

    example_data_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=1,
                                        shuffle=False,
                                        )
    perm = losses_idx[len(losses_idx) - int(len(losses_idx) * isolation_ratio):]

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        img = img.squeeze()
        target = target.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        # Filter the examples corresponding to losses_idx
        if idx in perm:
            isolation_examples.append((img, target))
            cnt += 1
        else:
            other_examples.append((img, target))

    # Save data
    if opt.save:
        os.makedirs(opt.isolate_data_root, exist_ok=True)
        data_path_isolation = os.path.join(opt.isolate_data_root, "{}_prune_isolation{}%_examples.npy".format(opt.model_name,
                                                                                             isolation_ratio * 100))
        data_path_other = os.path.join(opt.isolate_data_root, "{}_prune_other{}%_examples.npy".format(opt.model_name,
                                                                                             100 - isolation_ratio * 100))
        #print("ISE Last ", isolation_examples[-1])
        if os.path.exists(data_path_isolation):
            raise ValueError('isolation data already exists')
        else:
            isolation_examples_save = np.empty(len(isolation_examples),dtype=object)
            isolation_examples_save[:] = isolation_examples
            np.save(data_path_isolation, isolation_examples_save)
            other_examples_save = np.empty(len(other_examples),dtype=object)
            other_examples_save[:] = other_examples            
            np.save(data_path_other, other_examples_save)

    print('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    print('Finish collecting {} other examples: '.format(len(other_examples)))

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(model, neuron):
    state_dict = model.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    model.load_state_dict(state_dict)


def evaluate_by_number(opt, model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    cur_max = float('-inf')
    best_val = 0
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        acc_clean, acc_bd = test(opt, clean_loader, poison_loader, model, criterion)
        print('Start: {:.2f}, Layer: {}, Neuron Idx: {}, Value: {}'.format(start, layer_name, neuron_idx, value))
        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, value, acc_bd[2], acc_bd[0], acc_clean[2], acc_clean[0]))
        loss_diff = acc_bd[2] - acc_clean[2]
        cur_max = max(cur_max, loss_diff)
        is_best = (cur_max == loss_diff)
        if is_best:
          best_val = start
        save_checkpoint({
            'epoch': start,
            'state_dict': model.state_dict(),
            'clean_acc': acc_clean[0],
            'bad_acc': acc_bd[0],
        }, start, is_best, opt)
    return results, best_val


def evaluate_by_threshold(opt, model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    cur_max = float('-inf')
    best_threshold = 0
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        acc_clean, acc_bd = test(opt, clean_loader, poison_loader, model, criterion)
        print('Start: {:.2f}, Layer: {}, Neuron Idx: {}, Threshold: {}'.format(start, layer_name, neuron_idx, threshold))
        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, acc_bd[2], acc_bd[0], acc_clean[2], acc_clean[0]))
        loss_diff = acc_bd[2] - acc_clean[2]
        cur_max = max(cur_max, loss_diff)
        is_best = (cur_max == loss_diff)
        if is_best:
          best_threshold = threshold
        save_checkpoint({
            'epoch': threshold,
            'state_dict': model.state_dict(),
            'clean_acc': acc_clean[0],
            'bad_acc': acc_bd[0],
        }, threshold, is_best, opt)
    return results, best_threshold

def test(opt, test_clean_loader, test_bad_loader, model, criterion):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model(img)
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
            output = model(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    return acc_clean, acc_bd

def prune(opt):
    # Load models
    print('----------- Network Initialization --------------')

    model_directory = opt.mask_optimization_checkpoint_root + "/"
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

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    test_clean_loader, test_bad_loader = get_test_loader(opt)
    
    print('----------- Pruning Initialization --------------')
    mask_values = read_data(os.path.join(opt.mask_score_root, 'mask_values.txt'))
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model, criterion)
    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bad[0], acc_bad[2]))

    if opt.pruning_by == 'threshold':
        results, best = evaluate_by_threshold(opt, model, mask_values, opt.pruning_max, opt.pruning_step, criterion, test_clean_loader, test_bad_loader)
    else:
        results, best = evaluate_by_number(opt, model, mask_values, opt.pruning_max, opt.pruning_step, criterion, test_clean_loader, test_bad_loader)
    file_name = os.path.join(opt.log_root, 'pruning_by_{}.txt'.format(opt.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)   

    pruned_model, pruned_checkpoint_epoch = select_model(dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=opt.neuron_pruning_checkpoint_root +"/" + opt.model_name + "_prune_" + str(best) + ".tar",
                                n_classes=opt.num_class,
                                norm_layer=opt.norm_layer)

    pruned_model.to(opt.device)
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

    return poisoned_data, pruned_model    


def save_checkpoint(state, threshold, is_best, opt):
    if is_best:
        os.makedirs(opt.neuron_pruning_checkpoint_root, exist_ok=True)
        filepath = os.path.join(opt.neuron_pruning_checkpoint_root, opt.model_name + r'_prune_{}.tar'.format(threshold))
        torch.save(state, filepath)
    print('[info] Finish saving the model')


def main():
    print('----------- Prune model -----------')
    opt = get_arguments().parse_args()
    poisoned_data, model = prune(opt)

    print('----------- Calculate loss value per example -----------')
    losses_idx = compute_loss_value(opt, poisoned_data, model)

    print('----------- Collect isolation data -----------')
    isolate_data(opt, poisoned_data, losses_idx)

if (__name__ == '__main__'):
    main()