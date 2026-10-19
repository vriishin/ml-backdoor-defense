import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--isolation_model_root', type=str, default='./weight/ABL_results',
                        help='isolation model weights are saved here')
    parser.add_argument('--unlearning_root', type=str, default='./weight/ABL_results',
                        help='unlearning models weight are saved here')
    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--model_name', type=str, default='WRN-16-1', help='name of model')
    parser.add_argument('--tuning_checkpoint_root', type=str, default='./tuning_checkpoint',
                        help='tuning checkpoints are saved here')
    parser.add_argument('--finetuning_checkpoint_root', type=str, default='./finetuning_checkpoint',
                        help='tuning checkpoints are saved here')
    parser.add_argument('--unlearning_checkpoint_root', type=str, default='./unlearning_checkpoint',
                        help='unlearning checkpoints are saved here')
    parser.add_argument('--pretrained_epoch', type=int, default=5, help="epoch number for pretrained model")
    parser.add_argument('--isolate_data_root', type=str, default='./isolation_data',
                        help='path of isolated data')
    parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--tuning_epochs', type=int, default=10, help='number of tune epochs to run')
    parser.add_argument('--finetuning_ascent_model', type=str, default=True, help='whether finetuning model')
    parser.add_argument('--finetuning_epochs', type=int, default=50, help='number of finetuning epochs to run')
    parser.add_argument('--unlearning_epochs', type=int, default=5, help='number of unlearning epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_finetuning_init', type=float, default=0.1, help='initial finetuning learning rate')
    parser.add_argument('--lr_unlearning_init', type=float, default=5e-4, help='initial unlearning learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')
    parser.add_argument('--mask_isolation_ratio', type=float, default=0.01, help='ratio of isolation data for mask optimization')
    parser.add_argument('--gradient_ascent_type', type=str, default='Flooding', help='type of gradient ascent')
    parser.add_argument('--gamma', type=int, default=0.5, help='value of gamma')
    parser.add_argument('--flooding', type=int, default=0.5, help='value of flooding')

    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--interval', type=int, default=5, help='frequency of save model')

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--mask_opt_data_type', type=str, default="clean", choices=["clean", "custom", "random"], help='chooses what data to use for mask optimization')
    parser.add_argument('--finetuning_data_type', type=str, default="default", choices=["custom", "default"], help='chooses what data to use for fine tuning')
    parser.add_argument('--with_finetuned_model', type=bool, default=False, help='set as true when finetuning prior to mask optimization')
    parser.add_argument('--with_pruned_model', type=bool, default=False, help='chooses whether or not to use the pruned model for backdoor unlearning')
    parser.add_argument('--with_pruned_model_data', type=bool, default=False, help='chooses whether or not to use the post-pruning, separated data for backdoor unlearning')

    # backdoor attacks
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    # pruning
    parser.add_argument('--norm_layer', type=str, default=None, choices=[None, 'NoisyBatchNorm2d'], help='custom norm layer for pruning')
    parser.add_argument('--lr_mask_opt', type=float, default=0.2, help='the learning rate for mask optimization')
    parser.add_argument('--mask_opt_interval', type=int, default=25, help='frequency of save model')
    parser.add_argument('--mask_opt_sample_size', type=int, default=500, help='sample size of clean data used for mask optimization')
    parser.add_argument('--mask_opt_epochs', type=int, default=250, help='number of mask optimization epochs to run')
    parser.add_argument('--mask_optimization_checkpoint_root', type=str, default='./mask_optimization_checkpoint', help='mask optimization checkpoints are saved here')    
    parser.add_argument('--mask_pretrained_epoch', type=int, default=90, help="epoch number for masked model")    
    parser.add_argument('--mask_score_root', type=str, default='./mask_scores', help='path of mask scores')
    parser.add_argument('--anp_eps', type=float, default=0.1)
    parser.add_argument('--anp_steps', type=int, default=2)
    parser.add_argument('--anp_alpha', type=float, default=0.1)
    parser.add_argument('--pruning-by', type=str, default='threshold', choices=['number', 'threshold'])
    parser.add_argument('--pruning-max', type=float, default=0.90, help='the maximum number/threshold for pruning')
    parser.add_argument('--pruning-step', type=float, default=0.05, help='the step size for evaluating the pruning')
    parser.add_argument('--neuron_pruning_checkpoint_root', type=str, default='./pruned_model_checkpoint', help='pruned model checkpoint is saved here') 

    return parser
