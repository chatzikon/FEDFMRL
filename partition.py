import numpy as np
import torch
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, classes):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {classes[unq[i]]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    # print('mean:', np.mean(data_list))
    # print('std:', np.std(data_list))

    return net_cls_counts





def partition_data(y_train, partition, n_parties, classes, classes_percentage,
                   beta=0.4):

    n_train = y_train.shape[0]

    if partition == "random_split":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}




    elif partition == "noniid":
        min_size = 0
        min_require_size = 10

        if classes_percentage<1:
            no_class_client=[]
            min_size = -1
            min_require_size=0
            no_class_clients=round((1-classes_percentage)*n_parties)
            for i in range(no_class_clients):
                no_class_client.append(i%n_parties)


        K = len(classes)



        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(round(n_parties))]

            for k in range(K):

                if classes_percentage<1:
                    idx_batch_class= [[] for _ in range(round(n_parties))]

                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                #proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                #proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = np.random.dirichlet(np.repeat(beta, n_parties*classes_percentage))
                proportions = np.array([p * (len(idx_j) < N / (n_parties*classes_percentage)) for p, idx_j in zip(proportions, idx_batch)])

                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                if classes_percentage<1:
                    idx_batch_class = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_class, np.split(idx_k, proportions))]
                    min_size = 0
                    for i in range(len(no_class_client)):
                        idx_batch_class.insert(no_class_client[i], [])
                        no_class_client[i] = (no_class_client[-1] + i + 1) % n_parties
                    for l in range(len(idx_batch)):
                        idx_batch[l]+=idx_batch_class[l]
                else:
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])




        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    ss=0
    for i in range(len(net_dataidx_map)):
        ss+=len(net_dataidx_map[i])
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, classes)
    return ( net_dataidx_map, traindata_cls_counts)



def client_subset_creation(partition,trainSet,splits,n_clients,beta, batch_size, mode, common_dataset_size, classes_percentage,
                           dataset, datadir):


    if mode == 'distillation':


        total_len= int(len(trainSet)*(1-common_dataset_size))
        list_server=[int(len(trainSet)*common_dataset_size)]




    else:
        list_server = [int(len(trainSet) * 0.01)]
        total_len = int(len(trainSet) * 0.99)

    if partition=='random_split':

        train_set, valid_set, valid_dataset_server=random_split(n_clients, trainSet, splits, total_len, mode, list_server)



    else:

        train_set, valid_set, valid_dataset_server = noniid_split(total_len, mode, trainSet, list_server, n_clients, partition, beta,
                                                                  classes_percentage)



    train_loader_list, valid_loader_list, valid_loader_server=dataloader_creation(n_clients, train_set, valid_set, batch_size, mode, common_dataset_size,
                                                                                  valid_dataset_server)


    if dataset=='tinyimagenet':
        X_train, y_train, _, _ = load_tinyimagenet_data(datadir)
    else:
        X_train, y_train = trainSet.data, np.array(trainSet.targets)




    return train_loader_list, valid_loader_list, y_train, valid_loader_server, train_set, valid_set, valid_dataset_server

def dataloader_creation(n_clients, train_set, valid_set, batch_size, mode, common_dataset_size, valid_dataset_server):


    train_loader_list = []
    valid_loader_list = []

    net_dataidx_map_tr = {}
    net_dataidx_map_v = {}

    for client in range(n_clients):
        train_loader_list.append(
            torch.utils.data.DataLoader(train_set[client], batch_size=batch_size, shuffle=True))
        valid_loader_list.append(
            torch.utils.data.DataLoader(valid_set[client], batch_size=batch_size, shuffle=False))

        net_dataidx_map_tr[client] = np.array(train_loader_list[client].dataset.indices)
        net_dataidx_map_v[client] = np.array(valid_loader_list[client].dataset.indices)

    if mode == 'distillation':

        if common_dataset_size > 0:

            valid_loader_server = torch.utils.data.DataLoader(valid_dataset_server, batch_size=batch_size,
                                                              shuffle=False)


        else:

            valid_loader_server = 0

    else:

        valid_loader_server = torch.utils.data.DataLoader(valid_dataset_server, batch_size=batch_size, shuffle=False)

    return train_loader_list, valid_loader_list, valid_loader_server








def random_split(n_clients, trainSet, splits, total_len, mode, list_server):


    split_tr = []
    split_ev = []

    for i in range(n_clients):
        if np.mod(i, 2) == 0:
            split_tr.append(int(np.floor(len(trainSet) * splits[i])))
            split_ev.append(int(np.floor(len(trainSet) * splits[n_clients + i])))
        else:
            split_tr.append(int(np.ceil(len(trainSet) * splits[i])))
            split_ev.append(int(np.ceil(len(trainSet) * splits[n_clients + i])))


    if sum(split_tr) > (total_len * 0.9):
        split_tr[-1] = int(split_tr[-1] - (sum(split_tr) - total_len * 0.9))
    elif sum(split_tr) < (total_len * 0.9):
        split_tr[-1] = int(split_tr[-1] - (sum(split_tr) - total_len * 0.9))

    if sum(split_ev) > (total_len * 0.1):
        split_ev[-1] = int(split_ev[-1] - (sum(split_ev) - total_len * 0.1))
    elif sum(split_ev) < (total_len * 0.1):
        split_ev[-1] = int(split_ev[-1] - (sum(split_ev) - total_len * 0.1))





    if mode == 'distillation':

        if list_server[0]>0:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        split_tr + split_ev+list_server,
                                                        generator=torch.Generator().manual_seed(0))

        else:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        split_tr + split_ev,
                                                        generator=torch.Generator().manual_seed(0))

    else:

        tot_dataset = torch.utils.data.random_split(trainSet,
                                                    split_tr + split_ev + list_server,
                                                    generator=torch.Generator().manual_seed(0))

    train_set = []
    valid_set = []

    if mode == 'distillation':

        if list_server[0]>0:

            valid_dataset_server = tot_dataset[-1]
            temp_dataset = tot_dataset[:-1]

        else:

            valid_dataset_server = 0
            temp_dataset = tot_dataset

    else:

        valid_dataset_server = tot_dataset[-1]
        temp_dataset = tot_dataset[:-1]




    for i in range(n_clients):
        train_set.append(temp_dataset[i])
        valid_set.append(temp_dataset[n_clients + i])




    return train_set, valid_set, valid_dataset_server





def noniid_split(total_len, mode, trainSet, list_server, n_clients, partition, beta, classes_percentage):

    total_len = [total_len]

    if mode == 'distillation':

        if list_server[0] > 0:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        total_len + list_server,
                                                        generator=torch.Generator().manual_seed(0))

        else:

            tot_dataset = torch.utils.data.random_split(trainSet,
                                                        total_len,
                                                        generator=torch.Generator().manual_seed(0))


    else:

        tot_dataset = torch.utils.data.random_split(trainSet,
                                                    total_len + list_server,
                                                    generator=torch.Generator().manual_seed(0))

    if mode == 'distillation':

        if list_server[0] > 0:

            valid_dataset_server = tot_dataset[-1]
            temp_dataset = tot_dataset[0]

        else:

            valid_dataset_server = 0
            temp_dataset = tot_dataset[0]

    else:

        valid_dataset_server = tot_dataset[-1]
        temp_dataset = tot_dataset[0]


    temp = np.array(trainSet.targets)
    y_train = temp[np.array(temp_dataset.indices)]

    #classes=random.sample(trainSet.classes,int(classes_percentage*len(trainSet.classes)))
    classes = trainSet.classes

    net_dataidx_map_init, traindata_cls_counts = partition_data(y_train, partition, n_clients, classes, classes_percentage,
                                                                beta=beta)

    net_dataidx_map = {}

    for i in range(len(net_dataidx_map_init)):
        net_dataidx_map[i]=[]




    for i in range(len(net_dataidx_map_init)):
        for j in range(len(net_dataidx_map_init[i])):
            net_dataidx_map[i].append(temp_dataset.indices[net_dataidx_map_init[i][j]])


    train_set = []
    valid_set = []

    for client in range(n_clients):
        idxs = np.random.permutation(net_dataidx_map[client])
        #idxs = net_dataidx_map[client]
        delimiter = int(np.ceil(0.9 * len(idxs)))

        train_set.append(torch.utils.data.Subset(trainSet, idxs[:delimiter]))
        valid_set.append(torch.utils.data.Subset(trainSet, idxs[delimiter:]))


        # import collections
        #
        # trainset = train_set[client]
        # valset=valid_set[client]
        # targets_np = np.array(trainset.dataset.targets)
        # inds_np=np.concatenate((trainset.indices, valset.indices))
        #
        # targets = targets_np[inds_np]
        #
        # train_counter = collections.Counter(targets)
        # classes = trainset.dataset.classes
        #
        # for i, c in enumerate(classes):
        #     print(f"{c}: {train_counter[i]} images")


    return train_set, valid_set, valid_dataset_server


