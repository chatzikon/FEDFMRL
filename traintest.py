import numpy as np
import torch
import torchnet as tnt
import torch.nn.functional as F
import torchvision.models
from torch.autograd import Variable
from VGG import vgg
from resnet import resnet
from resnet_cifar import *
from utils import *
import torch.optim as optim
import time
from aggregation import *
import torch.nn as nn
import math
from partition import *
from exemplar import *
from torchvision import transforms
import random
import copy
from resnet_cifar_moon import *



def create_layer_optimizers(model,lr, momentum, weight_decay):

    l3_params = []
    l2_params = []
    l1_params = []

    for name, param in model.named_parameters():

        if name.startswith('linear'):
            pass

        elif name.startswith("layer3"):

            l3_params.append(param)

        elif name.startswith("layer2"):

            l3_params.append(param)
            l2_params.append(param)


        else:

            l3_params.append(param)
            l2_params.append(param)
            l1_params.append(param)

    optimizer_l1 = optim.SGD(l1_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_l2 = optim.SGD(l2_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_l3 = optim.SGD(l3_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    layer_optimizers = [optimizer_l1, optimizer_l2, optimizer_l3]


    return layer_optimizers



def communication_round_train(writer, n_rounds, n_clients, n_classes, normalization, mod, depth, momentum, weight_decay, lr, t_round, save_path,
                              mode, train_loader_list, valid_loader_list, total_epochs, count,coef_t,coef_d, lr_next, checkpoint_prev, extracted_logits,
                              valid_loader_server, common_dataset_size,
                              path, alphamix_global, alpha_cap, inverse, case_alpha, alpha_normalization, closed_form_approximation, alpha_opt, alpha_hyperparams,
                              nclients_div, multiloss, scale, multiloss_type, alpha_wd_factor,
                              SoTA_comp, poisoned_classes, modified_classes, poisoned_clients, dataset):








    best_round=0
    best_prec=0

    if mode == 'traditional':
        valid_loader_server = valid_loader_list[-1]







    per_task_soft_activation=[]
    per_round_soft_activation=0


    test_r_loss=[]
    test_r_prec=[]

    for round_n in range(n_rounds):



        per_client_output_avg = []
        per_client_layers_avg = []
        per_client_grads_avg = []
        per_client_target_avg=[]
        per_client_activation_avg = []
        per_client_tot_output_avg = []


        per_client_act_avg=[]

        if round_n == 0:
            list_name = 0
            soft_logits = [0]
            soft_logits_l=[0]



        test_c_loss=[]
        test_c_prec=[]



        for client in range(n_clients):






            per_client_output_avg, per_client_act_avg, per_client_layers_avg, per_client_grads_avg, per_client_target_avg, per_client_activation_avg, per_client_tot_output_avg, \
                list_name, epoch,  writer, test_c_loss, test_c_prec= \
                client_train(per_client_output_avg, per_client_act_avg, per_client_layers_avg, per_client_grads_avg, per_client_target_avg, per_client_activation_avg, per_client_tot_output_avg, client, round_n, mod, normalization,
                             depth,  lr, momentum, weight_decay,
                             writer, save_path, t_round, lr_next, train_loader_list, valid_loader_list, total_epochs,
                             n_classes, mode, coef_t,coef_d, list_name, soft_logits, soft_logits_l, valid_loader_server,
                             common_dataset_size,  path, alphamix_global, multiloss, scale, multiloss_type,
                             test_c_loss, test_c_prec, poisoned_classes, modified_classes, poisoned_clients, dataset)











        if mode=='traditional':

            if mod == 'vgg':
                model = vgg(normalization, n_classes, depth=depth)

            elif mod == 'resnet':
                # if dataset=='tinyimagenet':
                #     model = torchvision.models.resnet50(pretrained=True)
                #     model.fc = nn.Linear(model.fc.in_features, n_classes)
                # else:
                    model = resnet(n_classes=n_classes, depth=depth)


            model.cuda()



            model=load_model_federated_traditional(model, list_name, n_clients,path)






        elif mode=='distillation':







            if common_dataset_size==0:

                soft_logits=federated_distillation_aggregation(per_client_output_avg,per_client_activation_avg,per_client_tot_output_avg, n_classes, n_clients)






            else:


                per_client_output_avg_np = np.array(per_client_output_avg)

                if multiloss:




                    per_client_layers_avg_np=[]

                    for l in range(3):

                        temp=[]

                        for c in range(n_clients):

                            temp.append(per_client_layers_avg[c][l])


                        per_client_layers_avg_np.append(np.array(temp))

                    per_client_layers_avg_np.append(per_client_layers_avg[0][-1])












                if alphamix_global:
                    ####alphamix global
                    per_client_grads_avg_np = np.array(per_client_grads_avg)


                    if  alpha_opt:

                        if SoTA_comp:

                            # if depth == 11:
                            #     model_srv_alpha = ResNet11(n_classes)
                            # else:
                            model_srv_alpha = resnet(n_classes=n_classes, depth=56)
                        else:
                            model_srv_alpha =torchvision.models.resnet50(pretrained=False, progress=True)

                            model_srv_alpha.fc = torch.nn.Linear(model_srv_alpha.fc.in_features,n_classes)

                        model_srv_alpha.cuda()





                        optimizer_srv_a = optim.SGD(model_srv_alpha.parameters(), lr=lr_next[0], momentum=momentum,
                                              weight_decay=alpha_wd_factor)

                        if round_n > 0:
                            checkpoint_alpha = load_checkpoint_srv(save_path)
                            best_prec = checkpoint_alpha['best_prec']
                            model_srv_alpha.load_state_dict(checkpoint_alpha['state_dict'], strict=False)

                            optimizer_srv_a.load_state_dict(checkpoint_alpha['optimizer'])



                        model_srv_alpha.cuda()





                    else:

                        model_srv_alpha=0
                        optimizer_srv_a=0

                    soft_logits, model_srv_alpha, optimizer_srv_a= alpha_calculation(alpha_cap,per_client_output_avg_np,
                                                       per_client_output_avg_np, per_client_grads_avg_np,
                                                      inverse, case_alpha, alpha_normalization, closed_form_approximation, alpha_opt, alpha_hyperparams,
                                                     valid_loader_server,  model_srv_alpha, optimizer_srv_a,
                                                        nclients_div,  SoTA_comp)




                    if multiloss:

                        soft_logits_l = []

                        for l in range(3):
                            soft_logits_l.append(np.mean(per_client_layers_avg_np[l], axis=0))

                        soft_logits_l.append(per_client_layers_avg_np[-1])


                    if alpha_opt:



                        loss_server, prec_server= test(model_srv_alpha, valid_loader_server, alpha_opt, SoTA_comp)
                        prec_server = float(prec_server)
                        is_best = prec_server > best_prec
                        best_prec = max(prec_server, best_prec)

                        writer.add_scalar("Loss/valid_server", loss_server, round_n)
                        writer.add_scalar("Acc/valid_server", prec_server, round_n)
                        writer.flush()


                        checkpoint_prev=save_checkpoint_srv({
                            'epoch': round_n + 1,
                            'state_dict': model_srv_alpha.state_dict(),
                            'best_prec': best_prec,
                            'optimizer': optimizer_srv_a.state_dict(),
                        }, is_best, str(count) + '_' + str(round_n + 1), checkpoint_prev, n_clients, filepath=save_path)



                else:

                    soft_logits = np.mean(per_client_output_avg_np, axis=0)

                    if multiloss:

                        soft_logits_l = []

                        for l in range(3):
                            soft_logits_l.append(np.mean(per_client_layers_avg_np[l], axis=0))

                        soft_logits_l.append(per_client_layers_avg_np[-1])













            model = 0












        if mode=='traditional':



            loss_server, prec_server = test(model, valid_loader_server, False, False)

            if prec_server > best_prec:
                is_best = True
                best_prec = prec_server
                best_round = round_n + 1
            else:
                is_best = False

            writer.add_scalar("Loss/valid_server", loss_server, round_n)
            writer.add_scalar("Acc/valid_server", prec_server, round_n)

            save_checkpoint_srv({
                'epoch': round_n + 1,
                'state_dict': model.state_dict(),
                'best_prec': prec_server,
                'optimizer': 0,
            }, is_best, str(count)+'_'+str(round_n + 1),  checkpoint_prev, n_clients, filepath=save_path)

            print('The precision of the server is ' + str(prec_server))

            model.cpu()

            writer.flush()








    return best_prec, best_round, model, lr_next, round_n, test_r_loss, test_r_prec





def client_train(per_client_output_avg, per_client_act_avg, per_client_layers_avg, per_client_grads_avg, per_client_target_avg, per_client_activation_avg,
                 per_client_tot_output_avg, client,round_n, mod,normalization,depth,lr, momentum, weight_decay,
                 writer, save_path, t_round, lr_next, train_loader_list, valid_loader_list, total_epochs, n_classes, mode, coef_t,coef_d,
                 list_name, soft_logits, soft_logits_l,  valid_loader_server, common_dataset_size,
                 path, alphamix_global, multiloss, scale, multiloss_type,
                 test_c_loss, test_c_prec, poisoned_classes, modified_classes, poisoned_clients, dataset):



    print('Training of Client ' + str(client) + ' in Round ' + str(round_n))




    if mod == 'vgg':
        model = vgg(normalization, n_classes, depth=depth)



    elif mod == 'resnet':


        if depth==11:
            model = ResNet11(n_classes)
        # if dataset=='tinyimagenet':
        #     model = torchvision.models.resnet50(pretrained=True)
        #     model.fc = nn.Linear(model.fc.in_features, n_classes)
        else:
            model= resnet(n_classes=n_classes, depth=depth)



















    if round_n == 0:

        if client == 0:
            list_name = getName(model)


        model.cuda()



        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)



        if multiloss and (multiloss_type=='b' or multiloss_type=='c'):


            layer_optimizers = create_layer_optimizers(model, lr, momentum, weight_decay)


        else:

            layer_optimizers=0









        epoch_init = 0

    else:



        epoch_init = round_n * t_round









        if mode == 'traditional':

            checkpoint = load_checkpoint_srv(save_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=lr_next[client], momentum=momentum,
                                  weight_decay=weight_decay)


        elif mode == 'distillation':

            checkpoint_l = load_checkpoint_local('local_model_' + str(client), save_path)
            model.load_state_dict(checkpoint_l['state_dict'], strict=False)


            model.cuda()




            optimizer = optim.SGD(model.parameters(), lr=lr_next[client], momentum=momentum,
                                  weight_decay=weight_decay)


            optimizer.load_state_dict(checkpoint_l['optimizer'])

            if multiloss and (multiloss_type == 'b' or multiloss_type == 'c'):

                layer_optimizers = create_layer_optimizers(model, lr_next[client], momentum, weight_decay)


            else:

                layer_optimizers = 0



            if multiloss and (multiloss_type=='b' or multiloss_type=='c'):
               for l in range(3):
                    layer_optimizers[l].load_state_dict(checkpoint_l['optimizer_layers'][l])


        if coef_d>0:

            ###distill
            distill_loss=model_distillation(valid_loader_server, epoch_init, model, optimizer, layer_optimizers, soft_logits, soft_logits_l, coef_d,
                                            multiloss, multiloss_type, scale)

            writer.add_scalar("Distill loss/train" + str(client), distill_loss, epoch_init)

    train_loader = train_loader_list[client]
    valid_loader = valid_loader_list[client]




    for epoch in range(epoch_init, epoch_init + t_round):

        if epoch in [int(total_epochs * 0.5), int(total_epochs * 0.75)]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        print('LR')
        print(optimizer.param_groups[0]['lr'])
        lr_next[client] = optimizer.param_groups[0]['lr']

        start_time = time.time()





        if epoch == epoch_init + t_round - 1:
            last_epoch = True
        else:
            last_epoch = False


        ###train
        train_loss, train_prec, output_avg, activation_avg, tot_output_avg = train(train_loader, epoch, model, optimizer, coef_t, soft_logits, last_epoch,
          mode, common_dataset_size, round_n, client, poisoned_classes, modified_classes, poisoned_clients, n_classes)






        valid_loss, valid_prec = test(model, valid_loader, False, False)

        writer.add_scalar("Loss/train" + str(client), train_loss, epoch)
        writer.add_scalar("Acc/train" + str(client), train_prec, epoch)
        writer.add_scalar("Loss/valid" + str(client), valid_loss, epoch)
        writer.add_scalar("Acc/valid" + str(client), valid_prec, epoch)

        writer.flush()


        if last_epoch:

            if common_dataset_size>0 and mode=='distillation' :

                if dataset=='tinyimagenet':
                    output_base = np.zeros((100000, n_classes))
                else:
                    output_base = np.zeros((50000, n_classes))





                act_base = 0



                if alphamix_global:

                    if dataset == 'tinyimagenet':
                        grads_base = np.zeros((100000, n_classes))
                        target_base=np.zeros((100000, n_classes))
                    else:
                        grads_base = np.zeros((50000, n_classes))
                        target_base = np.zeros((50000, n_classes))

                else:

                    grads_base=0
                    target_base=0




                ###feature extraction
                extracted_logits, extracted_layer_logits, extracted_grads, extracted_targets, extracted_act =logits_extraction(model, valid_loader_server,
                n_classes, output_base,act_base, grads_base, target_base, alphamix_global, multiloss, scale)



            if mode == 'traditional':
                save_model(model, list_name, client,path)









        elapsed_time = time.time() - start_time
        print('Elapsed time is ' + str(elapsed_time))


    if mode=='distillation' :

        if common_dataset_size > 0:

            per_client_output_avg.append(extracted_logits)

            if multiloss:
                per_client_layers_avg.append(extracted_layer_logits)



        else:

            per_client_output_avg.append(output_avg)


    if alphamix_global:

        per_client_grads_avg.append(extracted_grads)
        per_client_target_avg.append(extracted_targets)



    per_client_activation_avg.append(activation_avg)
    per_client_tot_output_avg.append(tot_output_avg)





    if mode=='distillation':

        if multiloss and (multiloss_type=='b' or multiloss_type=='c'):
            optimizer_layers_state_dict=[]
            for l in range(3):
                optimizer_layers_state_dict.append(layer_optimizers[l].state_dict())

            save_checkpoint_local({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_layers': optimizer_layers_state_dict,
            }, 'local_model_' + str(client), filepath=save_path)


        else:

            save_checkpoint_local({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'local_model_' + str(client), filepath=save_path)





    model.cpu()


    if multiloss and (multiloss_type == 'b' or multiloss_type == 'c'):
        for l in range(3):
            optimizer_to_cpu(layer_optimizers[l])

    optimizer_to_cpu(optimizer)
    writer.flush()


    return per_client_output_avg, per_client_act_avg, per_client_layers_avg,  per_client_grads_avg, per_client_target_avg, per_client_activation_avg, \
        per_client_tot_output_avg, list_name,epoch,  writer, test_c_loss, test_c_prec


def evaluate(test_loader, path, mod, normalization,n_classes,depth, alpha_opt, SoTA_comp, dataset):

    if mod == 'vgg':
        model = vgg(normalization, n_classes, depth=depth)

    elif mod == 'resnet':
        if depth==11:
            model = ResNet11(n_classes)
        else:
            # if dataset=='tinyimagenet':
            #     model = torchvision.models.resnet50(pretrained=True)
            #     model.fc = nn.Linear(model.fc.in_features, n_classes)
            # else:
                model = resnet(n_classes=n_classes, depth=depth)

    model.eval()

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    test(model, test_loader, alpha_opt, SoTA_comp)

    return

def model_distillation(train_loader, epoch, model, optimizer, optimizer_layers, soft_logits, soft_logits_l, coef,
                       multiloss, multiloss_type, scale):

    model.train()

    kl_loss=0
    m=0
    n=0






    soft_logits_ts=torch.from_numpy(soft_logits).float().cuda()

    soft_logits_l_ts = []

    if multiloss:
        for l in range(3):
            soft_logits_l_ts.append(torch.from_numpy(soft_logits_l[l]).float().cuda())

        soft_logits_l_ts.append(soft_logits_l[-1])






    for batch_idx, (data, target, index) in enumerate(train_loader):




        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        if multiloss and (multiloss_type=='b' or multiloss_type=='c'):

            for l in range(3):
                optimizer_layers[l].zero_grad()


        output, activation, out_l1, out_l2 = model(data)

        intermediate_output=[out_l1, out_l2, activation]


        loss, loss_i =knowledge_distillation(output, soft_logits_ts,soft_logits_l_ts, multiloss, intermediate_output,
                                    index,target,0,coef, scale)











        if multiloss:


            if multiloss_type == 'a':

                for l in range(3):

                    if isinstance(loss_i[l], list):
                        pass

                    else:
                        loss_i[l].backward(retain_graph=True)



                loss.backward()

                optimizer.step()





            elif multiloss_type == 'b':

                loss.backward(retain_graph=True)

                optimizer.step()

                for l in range(3):

                    if l < 2:

                         if isinstance(loss_i[l], list):
                             pass

                         else:
                            loss_i[l].backward(retain_graph=True)

                         optimizer_layers[l].step()




                    else:

                        if isinstance(loss_i[l], list):
                            pass

                        else:

                            loss_i[l].backward()

                        optimizer_layers[l].step()


            elif multiloss_type == 'c':

                loss.backward(retain_graph=True)

                optimizer.step()

                for l in reversed(range(3)):

                    if l > 0:

                        loss_i[l].backward(retain_graph=True)

                        optimizer_layers[l].step()




                    else:

                        loss_i[l].backward()

                        optimizer_layers[l].step()


        else:

            loss.backward()

            optimizer.step()




        del output
        del activation

        # print(float(batch_idx)+1e-12)
        log_interval = 47
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f})\n'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))



    return loss.item()


def knowledge_distillation(output, soft_logits_ts,soft_logits_l_ts,multiloss, intermediate_output,
                           index,target,coef_t,coef_d, scale):










    if multiloss:

        soft_labels_i=[]

        for l in range(3):
            soft_labels_i.append([])

        temp=np.zeros((len(index)), dtype=np.int32)



        for i in range(len(index)):




            if  index[i].item() in soft_logits_l_ts[-1]:

                temp[i]=list(np.where(soft_logits_l_ts[-1] == index[i].item())[0])[0]

            else:

                temp[i]=-1


        for l in range(3):



            soft_labels_i[l].append(soft_logits_l_ts[l][temp, :, :])





    soft_labels = soft_logits_ts[index, :]


    if multiloss:

        soft_labels_index = []
        for r in range(len(temp)):
            if temp[r]!=-1:
                soft_labels_index.append(r)


    else:

        soft_labels_index = []
        for r in range(len(index)):

            if torch.all(soft_logits_ts[index[r]]==0) :

                pass

            else:
                soft_labels_index.append(r)












    loss_mse = nn.MSELoss()








    loss = coef_t * F.cross_entropy(output, target).cuda() + coef_d * loss_mse(
            output, soft_labels).cuda()




    if multiloss:



            loss_i=[]

            for l in range(3):

                out_s=multi_scale_output_calculation(intermediate_output[l], scale).cuda()




                if len(soft_labels_i[l])==0:
                    loss_i.append([0])
                    print('ZEROLOSSSS')

                else:
                    soft_labels_t = torch.stack(soft_labels_i[l])
                    if len(soft_labels_t.size())!=len(out_s.size()):
                        soft_labels_t=soft_labels_t.squeeze(0)

                    loss_i.append(0.1*1/3*loss_mse(out_s, soft_labels_t).cuda())


    else:

        loss_i=0




    return loss, loss_i

def train(train_loader, epoch, model, optimizer,coef,soft_logits,last_epoch,mode, common_dataset_size, round_n, client,
          poisoned_classes, modified_classes, poisoned_clients, n_classes):


    model.train()
    train_acc = 0.
    data_sum = 0







    m = nn.Softmax(dim=1)

    if last_epoch and mode == 'distillation' :


        output_avg = []
        tot_output_avg = []
        activation_avg = []

        if n_classes==200:
            for i in range(n_classes):
                output_avg.append([])
                tot_output_avg.append([])
                activation_avg.append([])
        else:
            for i in range(len(train_loader.dataset.dataset.classes)):
                output_avg.append([])
                tot_output_avg.append([])
                activation_avg.append([])

    #common = set(train_loader.batch_sampler.sampler.indices) & set(train_loader.dataset.indices)
    for batch_idx, (data, target, index) in enumerate(train_loader):



        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        if client in poisoned_clients:
            for i in range(len(target)):
                if target[i] in poisoned_classes:
                    pos=poisoned_classes.index(target[i])
                    target[i]=modified_classes[pos]




        output, activation, _, _ = model(data)






        if last_epoch and mode=='distillation' and (common_dataset_size==0):






            output=m(output)

            activation_avg, output_avg, tot_output_avg = activations_calculation (output,activation,target,output_avg,
                                                                                  activation_avg, tot_output_avg)



        else:

            output_avg=0
            activation_avg=0
            tot_output_avg = 0



        if (round_n>0) and coef>0 :





            if isinstance(soft_logits, np.ndarray):
                soft_logits_ts = torch.from_numpy(soft_logits).float().cuda()
            else:
                soft_logits_ts = torch.stack(soft_logits)



            soft_logits_l_ts=0
            multiloss=False
            intermediate_output=0
            scale=0



            loss, _ = knowledge_distillation(output, soft_logits_ts, soft_logits_l_ts, multiloss, intermediate_output,
                                          index, target, 1, coef, scale)










        else:

           loss = F.cross_entropy(output, target).cuda()

        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        data_sum += data.size()[0]
        del output
        del activation

        # print(float(batch_idx)+1e-12)
        log_interval = 5
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item(), train_acc, data_sum,
                       100. * float(train_acc) / (float(data_sum))))

    if len(train_loader)==0:
        print("loss doesn't exist")
        loss = 0
        return loss, float(train_acc) / float(len(train_loader.sampler)), output_avg, activation_avg, tot_output_avg

    return loss.item(), float(train_acc) / float(len(train_loader.sampler)), output_avg, activation_avg, tot_output_avg






def test(model, test_loader, alpha_opt, SoTA_comp):
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    correct = 0
    with torch.no_grad():
        for data, target, index in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            output, activation, _, _ = model(data)
            loss = F.cross_entropy(output, target)
            test_loss.add(loss.item())  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            del output
            del activation

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss.item(), correct, len(test_loader.sampler),
            100. * float(correct) / len(test_loader.sampler)))
    return loss.item(), float(correct) / float(len(test_loader.sampler))



def logits_extraction(model,train_loader, n_classes, output_base,act_base,
                      grads_base, target_base, alphamix_global, multiloss, scale):




    model.eval()


    output = np.zeros((len(train_loader.dataset.indices), n_classes))








    inds = np.zeros((len(train_loader.dataset.indices),), dtype=int)

    if alphamix_global:

        output_grad = np.zeros((len(train_loader.dataset.indices), n_classes))
        target_ar = np.zeros(len(train_loader.dataset.indices), dtype=int)
        inds_grad = np.zeros((len(train_loader.dataset.indices),), dtype=int)

    k=0
    soft_label=[]

    if multiloss:
        soft_label_l1=[]
        soft_label_l2=[]
        soft_label_l3=[]

    grad_avg=[]

    for i in range(n_classes):
        soft_label.append([])

        if multiloss:
            soft_label_l1.append([])
            soft_label_l2.append([])
            soft_label_l3.append([])

        grad_avg.append([])





    if alphamix_global:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(alphamix_global):
        for batch_idx, (datat, target,index) in enumerate(train_loader):



            datat, target = datat.cuda(), target.cuda()


            datat, target= Variable(datat), Variable(target)


            out, activation, out_l1, out_l2=model(datat)


            if multiloss :

               out_l1_s = multi_scale_output_calculation(out_l1, scale)

               out_l2_s = multi_scale_output_calculation(out_l2, scale)

               out_l3_s = multi_scale_output_calculation(activation, scale)



               if batch_idx==0:
                   output_l1 = np.zeros((len(train_loader.dataset.indices), out_l1_s.size()[1], out_l1_s.size()[2]), dtype=np.float32)
                   output_l2 = np.zeros((len(train_loader.dataset.indices), out_l2_s.size()[1], out_l2_s.size()[2]), dtype=np.float32)
                   output_l3 = np.zeros((len(train_loader.dataset.indices), out_l3_s.size()[1], out_l3_s.size()[2]), dtype=np.float32)

            if alphamix_global:

                _, probs_sort_idxs = out.sort(descending=True)
                pred_1 = probs_sort_idxs[:, 0]

                loss = F.cross_entropy(out, pred_1)
                grads = torch.autograd.grad(loss, out)[0].data.cpu()



                output_grad[k:k + datat.shape[0], :] = grads.data.cpu().numpy()
                target_ar[k:k + datat.shape[0]] = target.cpu().numpy()
                inds_grad[k:k + datat.shape[0]] = index.cpu().numpy()










            output[k:k + datat.shape[0], :] = out.data.cpu().numpy()

            if multiloss:
                output_l1[k:k + datat.shape[0], :] = out_l1_s.data.cpu().numpy()
                output_l2[k:k + datat.shape[0], :] = out_l2_s.data.cpu().numpy()
                output_l3[k:k + datat.shape[0], :] = out_l3_s.data.cpu().numpy()

            inds[k:k + datat.shape[0]] = index.cpu().numpy()








            k = k + datat.shape[0]



    if alphamix_global:




        for i in range(len(inds_grad)):

              grads_base[inds_grad[i], :] = output_grad[i, :]
              target_base[inds_grad[i]] = target_ar[i]





    for i in range(len(inds)):

        output_base[inds[i], :] = output[i, :]


        if multiloss:

            output_base_layers=[output_l1,output_l2, output_l3,inds]


        else:

            output_base_layers=0




    return output_base, output_base_layers, grads_base, target_base, act_base









