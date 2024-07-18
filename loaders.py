
from partition import *
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from copy import copy

class CIFAR100_withIndex(datasets.CIFAR100):

    def __getitem__(self, index):
        img, label = super(CIFAR100_withIndex, self).__getitem__(index)

        return (img, label, index)


class CIFAR10_withIndex(datasets.CIFAR10):

    def __getitem__(self, index):
        img, label = super(CIFAR10_withIndex, self).__getitem__(index)

        return (img, label, index)



def loader_build(dataset,datadir,mode, partition,splits,n_clients,beta,batch_size, common_dataset_size, classes_percentage):

    transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if dataset=='cifar10':

        trainSet = CIFAR10_withIndex(root='./CifarTrainData', train=True,
                                                 download=True, transform=transform)


        trainSet_val= CIFAR10_withIndex(root='./CifarTrainData', train=True,
                                                 download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]))

        testSet = CIFAR10_withIndex(root='./CifarTrainData', train=False,
                                                download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]))


    elif dataset=='cifar100':

        trainSet = CIFAR100_withIndex(root='./CifarTrainData', train=True,
                                                download=True, transform=transform)

        trainSet_val = CIFAR100_withIndex(root='./CifarTrainData', train=True,
                                                    download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))


        testSet = CIFAR100_withIndex(root='./CifarTrainData', train=False,
                                                 download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif dataset=='tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainSet= dl_obj(datadir + '/train/', transform=transform_train)
        trainSet_val=dl_obj(datadir + '/train/', transform=transform_test)
        #testSet = dl_obj(datadir + '/test/', transform=transform_test)
        testSet = dl_obj(datadir + '/val/', transform=transform_test)


    if dataset=='cifar10':
        n_classes=10
    elif dataset=='cifar100':
        n_classes=100
    elif dataset=='tinyimagenet':
        n_classes=200


    test_loader = torch.utils.data.DataLoader(testSet)



    if mode == 'distillation':


            train_loader_list, valid_loader_list, _, _, train_set, valid_set, valid_dataset_server = \
                client_subset_creation(partition, trainSet, splits, n_clients , beta, batch_size, mode, common_dataset_size,
                                       classes_percentage, dataset, datadir)

            _, _, _, valid_loader_server,_,_,_ = \
                client_subset_creation(partition, trainSet_val, splits, n_clients , beta, batch_size, mode,common_dataset_size,
                                       classes_percentage, dataset, datadir)

    elif mode=='traditional':

            train_loader_list, valid_loader_list, _, _, train_set, valid_set, valid_dataset_server = \
                client_subset_creation(partition,trainSet,splits,n_clients,beta, batch_size, mode, common_dataset_size,
                                       classes_percentage, dataset, datadir)

            _, _, _, valid_loader_server, _, _, _ = \
                client_subset_creation(partition, trainSet_val, splits, n_clients , beta, batch_size, mode, common_dataset_size,
                                       classes_percentage, dataset, datadir)







    return train_loader_list, valid_loader_list, test_loader, n_classes, valid_loader_server, train_set, valid_set, valid_dataset_server,testSet
