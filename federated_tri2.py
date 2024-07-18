import numpy as np
import torch

from VGG import vgg
from resnet import resnet

import time
import random
from utils import *
from partition import *
from loaders import *
from torch.utils.tensorboard import SummaryWriter
from traintest import *
from aggregation import *
import os
from resnet_cifar import *
from resnet_cifar_moon import *

from torchvision import models


def print_cuda_info():
    print("Is cuda available?", torch.cuda.is_available())
    print("Is cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled? ", torch.backends.cudnn.enabled)
    print("Device count?", torch.cuda.device_count())
    print("Current device?", torch.cuda.current_device())
    print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))


def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)


def remove_unwanted(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            if ('test' in f):
                pass
            else:
                if 'pth.tar' in f:
                    os.remove(f)
            print(f)


def main(total_epochs, evaluate, path, splits, t_round, wd, normalization, n_clients, seed, count, lr, dataset,
         partition, mod, depth, mode, coef_t, coef_d,
         save_path, common_dataset_size, path_s,
         alphamix_global, alpha_cap, inverse, case_alpha, alpha_normalization, closed_form_approximation, alpha_opt,
         alpha_hyperparams, beta,
         nclients_div, multiloss, scale, multiloss_type, alpha_wd_factor, SoTA_comp, batch_size, classes_percentage,
         n_poisoned_classes, n_poisoned_clients, datadir):
    writer = SummaryWriter(log_dir='./' + save_path)
    print(splits)



    seed_everything(seed)

    print('Local epochs are ' + str(t_round))
    checkpoint_prev = 0

    lr_next = np.zeros(n_clients)

    (train_loader_list, valid_loader_list, test_loader, n_classes, valid_loader_server, train_set, valid_set,
     valid_dataset_server, testSet) = \
        loader_build(dataset, datadir, mode, partition, splits, n_clients, beta, batch_size, common_dataset_size,
                     classes_percentage)

    if evaluate == True:
        evaluate(test_loader, path, mod, normalization, n_classes, depth, alpha_opt, SoTA_comp, dataset)

    momentum = 0.9
    weight_decay = wd

    n_rounds = total_epochs // t_round

    if n_rounds * t_round < total_epochs:
        n_rounds += 1
        total_epochs = n_rounds * t_round

    extracted_logits = [0]

    if n_poisoned_clients > 0 and n_poisoned_classes > 0:

        clients_list = [i for i in range(n_clients)]
        classes_list = [i for i in range(n_classes)]

        poisoned_clients = random.sample(clients_list, n_poisoned_clients)
        classes_t = random.sample(classes_list, 2 * n_poisoned_classes)
        poisoned_classes = classes_t[0:len(classes_t) // 2]
        modified_classes = classes_t[len(classes_t) // 2:]

    else:

        poisoned_clients = []
        poisoned_classes = []
        modified_classes = []

    best_prec, best_round, model, lr_next, round_n, test_r_loss, test_r_prec = communication_round_train(writer,
                                                                                                         n_rounds,
                                                                                                         n_clients,
                                                                                                         n_classes,
                                                                                                         normalization,
                                                                                                         mod, depth,
                                                                                                         momentum,
                                                                                                         weight_decay,
                                                                                                         lr, t_round,
                                                                                                         save_path,
                                                                                                         mode,
                                                                                                         train_loader_list,
                                                                                                         valid_loader_list,
                                                                                                         total_epochs,
                                                                                                         count, coef_t,
                                                                                                         coef_d,
                                                                                                         lr_next,
                                                                                                         checkpoint_prev,
                                                                                                         extracted_logits,
                                                                                                         valid_loader_server,
                                                                                                         common_dataset_size,
                                                                                                         path_s,
                                                                                                         alphamix_global,
                                                                                                         alpha_cap,
                                                                                                         inverse,
                                                                                                         case_alpha,
                                                                                                         alpha_normalization,
                                                                                                         closed_form_approximation,
                                                                                                         alpha_opt,
                                                                                                         alpha_hyperparams,
                                                                                                         nclients_div,
                                                                                                         multiloss,
                                                                                                         scale,
                                                                                                         multiloss_type,
                                                                                                         alpha_wd_factor,
                                                                                                         SoTA_comp,
                                                                                                         poisoned_classes,
                                                                                                         modified_classes,
                                                                                                         poisoned_clients,
                                                                                                         dataset)

    writer.close()

    if (mode == 'traditional'):

        model.cuda()

        checkpoint = load_checkpoint(best_prec, str(count) + '_' + str(best_round), n_clients, save_path)

        model.load_state_dict(checkpoint['state_dict'])

        loss_server, prec_server = test(model, test_loader, False, False)

        save_checkpoint({
            'epoch': round_n + 1,
            'state_dict': model.state_dict(),
            'best_prec': prec_server,
            'variation': 0,
            'optimizer': 0,
        }, True, 'test_result_' + str(count),
            n_clients, filepath=save_path)

        print('The precision of the server at test set is ' + str(prec_server))


    else:

        valid_prec_clients = torch.zeros(n_clients)

        if mod == 'vgg':
            model = vgg(normalization, n_classes, depth=depth)



        elif mod == 'resnet':
            if depth == 11:
                model = ResNet11(n_classes)
            else:
                if dataset=='tinyimagenet':
                    model = torchvision.models.resnet50(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, n_classes)
                else:
                    model = resnet(n_classes=n_classes, depth=depth)

        model.cuda()

        f = open(save_path + "/client_model_results.txt", "w")

        for i in range(n_clients):
            checkpoint = load_checkpoint_local('local_model_' + str(i), save_path)
            model.load_state_dict(checkpoint['state_dict'])
            loss, prec = test(model, test_loader, False, False)
            f.write(str(prec) + '\n')
            valid_prec_clients[i] = prec

        f.close()

        prec_server = torch.mean(valid_prec_clients).item()
        variation = torch.std(valid_prec_clients).item()

        save_checkpoint({
            'epoch': round_n + 1,
            'state_dict': 0,
            'best_prec': prec_server,
            'variation': variation,
            'optimizer': 0,
        }, True, 'test_result_' + str(count),
            n_clients, filepath=save_path)

    model.cpu()

    if alpha_opt and alphamix_global:
        if SoTA_comp:
            if depth == 11:
                model_srv_alpha = ResNet11(n_classes)
            else:
                model_srv_alpha = resnet(n_classes=n_classes, depth=depth)
        else:
            model_srv_alpha = torchvision.models.resnet50(pretrained=True, progress=True)
            model_srv_alpha.fc = torch.nn.Linear(model_srv_alpha.fc.in_features, n_classes)

        model_srv_alpha.cuda()

        checkpoint_alpha = load_checkpoint_srv(save_path)
        model_srv_alpha.load_state_dict(checkpoint_alpha['state_dict'])

        loss_server, prec_server = test(model_srv_alpha, test_loader, True, SoTA_comp)

        save_checkpoint({
            'epoch': round_n + 1,
            'state_dict': model.state_dict(),
            'best_prec': prec_server,
            'variation': 0,
            'optimizer': 0,
        }, True, 'test_result_alpha_server_' + str(count),
            n_clients, filepath=save_path)

        print('The precision of the alpha server at test set is ' + str(prec_server))

    remove_unwanted(save_path)

    return prec_server


if __name__ == '__main__':

    path_s = '2'

    torch.cuda.set_device(int(path_s))

    torch.cuda.synchronize()

    print_cuda_info()

    argument_dict = {}

    argument_dict['path'] = ''
    argument_dict['wd_factor'] = 1e-5
    argument_dict['normalization'] = 'normal'
    argument_dict['lr'] = 0.01
    argument_dict['mod'] = 'resnet'

    count = 0
    iters = 1

    ###1####

    argument_dict['partition'] = 'random_split'
    argument_dict['common_dataset_size'] = 0.2

    argument_dict['depth'] = 56

    argument_dict['batch_size'] = 512

    argument_dict['clients'] = 10
    argument_dict['t_round'] = 1
    argument_dict['epochs'] = 3







    ##fld_alphamix_isolated

    argument_dict['alpha_cap'] = 0.8

    argument_dict['dataset'] = 'tinyimagenet'
    argument_dict['mode'] = 'distillation'
    argument_dict['coef_t'] = 0
    argument_dict['coef_d'] = 0

    argument_dict['alphamix_global'] = False
    argument_dict['alpha_normalization'] = 'none'
    argument_dict['case_alpha'] = 'b'
    argument_dict['inverse'] = 'mix1'
    argument_dict['closed_form_approximation'] = False
    argument_dict['alpha_opt'] = True
    argument_dict['alpha_learning_iters'] = 5
    argument_dict['alpha_grads_div'] = 1
    argument_dict['alpha_learning_rate'] = 1e-4
    argument_dict['alpha_clf_coef'] = 1
    argument_dict['alpha_l2_coef'] = 0.01
    argument_dict['alpha_grads_div'] = 20
    argument_dict['alpha_wd_factor'] = 1e-5
    argument_dict['nclients_div'] = False
    argument_dict['SoTA_comp'] = False

    argument_dict['multiloss'] = False
    argument_dict['multiloss_type'] = 'b'
    argument_dict['scale'] = 1
    argument_dict['beta'] = 0.5

    argument_dict['datadir'] = '../tiny-imagenet-200'

    ##new age##
    argument_dict['classes_percentage'] = 1
    # classes_percentage=[0.4,0.6,0.8,1,0.2]
    number_of_poisoned_classes = [0]
    number_of_poisoned_clients = [0]

    argument_dict['alpha_hyperparams'] = [argument_dict['alpha_learning_rate'],
                                          argument_dict['alpha_learning_iters'],
                                          argument_dict['alpha_clf_coef'], argument_dict['alpha_l2_coef'],
                                          argument_dict['alpha_grads_div']]

    for i in range(len(number_of_poisoned_classes)):
        for k in range(len(number_of_poisoned_clients)):

            argument_dict['number_of_poisoned_classes'] = number_of_poisoned_classes[i]
            argument_dict['number_of_poisoned_clients'] = number_of_poisoned_clients[k]

            for j in range(iters):
                splits = create_splits(argument_dict)
                save_path = create_save_folder(argument_dict, path_s)

                main(argument_dict['epochs'], False, argument_dict['path'], splits, argument_dict['t_round'],
                     argument_dict['wd_factor'], argument_dict['normalization']
                     , argument_dict['clients'], j, count, argument_dict['lr'], argument_dict['dataset'],
                     argument_dict['partition'], argument_dict['mod'],
                     argument_dict['depth'], argument_dict['mode'], argument_dict['coef_t'],
                     argument_dict['coef_d'],
                     save_path,
                     argument_dict['common_dataset_size'], path_s,
                     argument_dict['alphamix_global'], argument_dict['alpha_cap'], argument_dict['inverse'],
                     argument_dict['case_alpha'],
                     argument_dict['alpha_normalization'], argument_dict['closed_form_approximation'],
                     argument_dict['alpha_opt'], argument_dict['alpha_hyperparams'],
                     argument_dict['beta'], argument_dict['nclients_div'],
                     argument_dict['multiloss'], argument_dict['scale'],
                     argument_dict['multiloss_type'], argument_dict['alpha_wd_factor'],
                     argument_dict['SoTA_comp'], argument_dict['batch_size'], argument_dict['classes_percentage'],
                     argument_dict['number_of_poisoned_classes'], argument_dict['number_of_poisoned_clients'],
                     argument_dict['datadir'])
                count += 1





    # ###fld_trad

    # argument_dict['alpha_cap'] = 0.8
    #
    # argument_dict['dataset'] = 'tinyimagenet'
    # argument_dict['mode'] = 'distillation'
    # argument_dict['coef_t'] = 0
    # argument_dict['coef_d'] = 1
    #
    # argument_dict['alphamix_global'] = False
    # argument_dict['alpha_normalization'] = 'none'
    # argument_dict['case_alpha'] = 'b'
    # argument_dict['inverse'] = 'mix1'
    # argument_dict['closed_form_approximation'] = False
    # argument_dict['alpha_opt'] = True
    # argument_dict['alpha_learning_iters'] = 5
    # argument_dict['alpha_grads_div'] = 1
    # argument_dict['alpha_learning_rate'] = 1e-4
    # argument_dict['alpha_clf_coef'] = 1
    # argument_dict['alpha_l2_coef'] = 0.01
    # argument_dict['alpha_grads_div'] = 20
    # argument_dict['alpha_wd_factor'] = 1e-5
    # argument_dict['nclients_div'] = False
    # argument_dict['SoTA_comp'] = False
    #
    # argument_dict['multiloss'] = False
    # argument_dict['multiloss_type'] = 'b'
    # argument_dict['scale'] = 1
    # argument_dict['beta'] = 0.5
    #
    # argument_dict['datadir'] = '../tiny-imagenet-200'
    #
    # ##new age##
    # argument_dict['classes_percentage'] = 1
    # # classes_percentage=[0.4,0.6,0.8,1,0.2]
    # number_of_poisoned_classes = [0]
    # number_of_poisoned_clients = [0]
    #
    # argument_dict['alpha_hyperparams'] = [argument_dict['alpha_learning_rate'],
    #                                       argument_dict['alpha_learning_iters'],
    #                                       argument_dict['alpha_clf_coef'], argument_dict['alpha_l2_coef'],
    #                                       argument_dict['alpha_grads_div']]
    #
    # for i in range(len(number_of_poisoned_classes)):
    #     for k in range(len(number_of_poisoned_clients)):
    #
    #         argument_dict['number_of_poisoned_classes'] = number_of_poisoned_classes[i]
    #         argument_dict['number_of_poisoned_clients'] = number_of_poisoned_clients[k]
    #
    #         for j in range(iters):
    #             splits = create_splits(argument_dict)
    #             save_path = create_save_folder(argument_dict, path_s)
    #
    #             main(argument_dict['epochs'], False, argument_dict['path'], splits, argument_dict['t_round'],
    #                  argument_dict['wd_factor'], argument_dict['normalization']
    #                  , argument_dict['clients'], j, count, argument_dict['lr'], argument_dict['dataset'],
    #                  argument_dict['partition'], argument_dict['mod'],
    #                  argument_dict['depth'], argument_dict['mode'], argument_dict['coef_t'],
    #                  argument_dict['coef_d'],
    #                  save_path,
    #                  argument_dict['common_dataset_size'], path_s,
    #                  argument_dict['alphamix_global'], argument_dict['alpha_cap'], argument_dict['inverse'],
    #                  argument_dict['case_alpha'],
    #                  argument_dict['alpha_normalization'], argument_dict['closed_form_approximation'],
    #                  argument_dict['alpha_opt'], argument_dict['alpha_hyperparams'],
    #                  argument_dict['beta'], argument_dict['nclients_div'],
    #                  argument_dict['multiloss'], argument_dict['scale'],
    #                  argument_dict['multiloss_type'], argument_dict['alpha_wd_factor'],
    #                  argument_dict['SoTA_comp'], argument_dict['batch_size'], argument_dict['classes_percentage'],
    #                  argument_dict['number_of_poisoned_classes'], argument_dict['number_of_poisoned_clients'],
    #                  argument_dict['datadir'])
    #             count += 1
    #


