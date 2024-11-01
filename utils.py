import numpy as np
import torch
import os
import shutil
from datetime import datetime
import math
from torch.autograd import Variable
import torch.nn.functional as F
from VGG import vgg
from resnet import *
import torch.nn as nn
def learn_alpha(model, org_embed,  labels, anchor_embed, alpha,  alpha_learning_iters, alpha_learning_rate,alpha_clf_coef,
                alpha_l2_coef, valid_loader_server, optimizer, alpha_grads_div,
                SoTA_comp):

    if SoTA_comp==False:
        n_classes = model.fc.out_features
    else:
        n_classes=model.linear.out_features


    non_empty_mask = labels.abs().sum(dim=1).bool()

    alpha_nz=alpha[non_empty_mask,:]





    org_embed_nz=org_embed[non_empty_mask,:]
    anchor_embed_nz=anchor_embed[non_empty_mask,:]


    loss_func = torch.nn.CrossEntropyLoss()


    model.train()



    for i in range(alpha_learning_iters):



        l = alpha_nz
        l = torch.autograd.Variable(l.cuda(), requires_grad=True)
        opt = torch.optim.Adam([l], lr=alpha_learning_rate / (
            1. if i < alpha_learning_iters * 2 / 3 else 10.))
        e = org_embed_nz.cuda()
        c_e = anchor_embed_nz.cuda()
        embedding_mix = (1 - l) * e + l * c_e

        inds = torch.zeros(len(valid_loader_server.sampler.data_source.indices), dtype=torch.int64)




        if n_classes>100:
            output_base = torch.zeros(100000, n_classes, dtype=torch.float64)
            target_base = torch.zeros(100000, n_classes, dtype=torch.int64)
        else:
            output_base = torch.zeros(50000, n_classes, dtype=torch.float64)
            target_base = torch.zeros(50000, n_classes, dtype=torch.int64)
        output = torch.zeros(len(valid_loader_server.dataset.indices), n_classes)
        target_ar = torch.zeros(len(valid_loader_server.dataset.indices), dtype=torch.int64)


        k=0


        temp_out = []

        for c in range(n_classes):
            temp_out.append([])


        for batch_idx, (data, target, index) in enumerate(valid_loader_server):

            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()


            out,activation, out_l1, out_l2=model(data)




            loss2 = loss_func(out, target).cuda()

            loss2.backward()

            for idx, param in enumerate(model.parameters()):
                param.grad=param.grad/alpha_grads_div

            optimizer.step()







            output[k:k + data.shape[0], :] = out.data

            inds[k:k + data.shape[0]] = index
            target_ar[k:k + data.shape[0]] = target
            k = k + data.shape[0]

            for j in range(len(inds)):
                output_base[inds[j], :] = output[j, :]
                target_base[inds[j]] = target_ar[j]






            output_base_nz = output_base[non_empty_mask, :].cuda()


            _, probs_sort_idxs = output_base_nz.sort(descending=True)
            labels_nz = probs_sort_idxs[:, 0]




        opt.zero_grad()








        l2_nrm = torch.norm(l, dim=1)









        #######define alpha server model loss




        clf_loss = loss_func(embedding_mix, labels_nz.cuda())

        loss1 = alpha_clf_coef * clf_loss + alpha_l2_coef * l2_nrm

        loss1.sum().backward()




        opt.step()





        l=torch.clamp(l, min=1e-8, max=1)

        alpha_nz = l.detach().cpu()



        del l, e, c_e, embedding_mix
        torch.cuda.empty_cache()

    alpha_f = torch.zeros(alpha.size(), dtype=torch.float)



    inds,indeces=inds.sort(descending=False)


    count = 0

    if n_classes>100:
        lim=100000
    else:
        lim=50000

    for i in range(lim):
        if inds[count] == i:


            alpha_f[i] = alpha_nz[count]
            count += 1
            if count == len(inds):
                break







    return alpha_f.cpu(), model, optimizer





def calculate_optimum_alpha(eps, lb_embedding, ulb_embedding, ulb_grads):
    z = (lb_embedding - ulb_embedding)  # * ulb_grads


    alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1) * ulb_grads / (z + 1e-8)

    return alpha


def generate_alpha(size, alpha_cap):
    if len(size)==2:
        alpha = torch.normal(
            mean=alpha_cap,
            std=alpha_cap,
            size=(size[0],size[1])).squeeze()
    elif len(size)==3:
        alpha = torch.normal(
            mean=alpha_cap,
            std=alpha_cap,
            size=(size[0], size[1],size[2])).squeeze()

    alpha[torch.isnan(alpha)] = 1

    return alpha

def getName(model):
    list_name = []

    for name in model.state_dict():
        list_name.append(name)

    return list_name

def optimizer_to_cpu(optim):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.cpu()
            if param._grad is not None:
                param._grad.data = param._grad.data.cpu()
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.cpu()
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.cpu()






def save_checkpoint(state, is_best, counter, n_clients, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath,
                                                                                   'model_best_test_acc_' + str(
                                                                                       counter) + '_' + str(
                                                                                       state['best_prec']) + '_' +'variation='+str(
                                                                                       state['variation'])+'_'+
                                                                                   '_n_clients=' + str(
                                                                                       n_clients) + '.pth.tar'))


def save_checkpoint_local(state, counter,  filepath):
    torch.save(state, os.path.join(filepath, str(counter)+ '_checkpoint.pth.tar'))



def save_checkpoint_srv(state, is_best, counter, checkpoint_prev, n_clients,  filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint_srv.pth.tar'))

    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint_srv.pth.tar'),
                        os.path.join(filepath, 'model_best_test_acc_' + str(
                            counter) + '_' + str(state['best_prec']) + '_'  + '_n_clients=' + str(n_clients) + '.pth.tar'))

        if checkpoint_prev != 0:
            os.remove(checkpoint_prev)

        checkpoint_prev= os.path.join(filepath, 'model_best_test_acc_' + str(
                            counter) + '_' + str(state['best_prec']) + '_'  + '_n_clients=' + str(n_clients) + '.pth.tar')

    return checkpoint_prev


def load_checkpoint_local(counter,filepath):
    if os.path.isfile(os.path.join(filepath,  str(counter)+ '_checkpoint.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath,  str(counter)+ '_checkpoint.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath,  str(counter)+ '_checkpoint.pth.tar'))
        print("=> loaded checkpoint '{}' ".format(os.path.join(filepath,  str(counter)+ '_checkpoint.pth.tar')))
    else:
        print("=> no checkpoint found at '{}'".format(
            os.path.join(filepath,  str(counter)+ '_checkpoint.pth.tar')))
    return checkpoint

def load_checkpoint_srv(filepath):
    if os.path.isfile(os.path.join(filepath, 'checkpoint_srv.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath, 'checkpoint_srv.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath, 'checkpoint_srv.pth.tar'))
        print("=> loaded checkpoint '{}' ".format(os.path.join(filepath, 'checkpoint_srv.pth.tar')))
    else:
        print("=> no checkpoint found at '{}'".format(
            os.path.join(filepath, 'checkpoint_srv.pth.tar')))
    return checkpoint


def load_checkpoint(best, counter,  n_clients, filepath):
    if os.path.isfile(os.path.join(filepath, 'model_best_test_acc_' + str(counter) + '_' + str(best) + '_'  +
                                             '_n_clients=' + str(n_clients) + '.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath,
                                                               'model_best_test_acc_' + str(counter) + '_' + str(
                                                                   best) + '_'   +
                                                               '_n_clients=' + str(n_clients) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath,
                                             'model_best_test_acc_' + str(counter) + '_' + str(best) + '_'  +
                                             '_n_clients=' + str(n_clients) + '.pth.tar'))
        print("=> loaded checkpoint '{}'  Prec1: {:f}".format(os.path.join(filepath, 'model_best_test_acc_' + str(
            counter) + '_' + str(best) + '_' +
                                                                           '_n_clients=' + str(n_clients) + '.pth.tar'),
                                                              best))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(filepath,
                                                                   'model_best_test_acc_' + str(counter) + '_' + str(
                                                                       best) + '_'   +
                                                                   '_n_clients=' + str(n_clients) + '.pth.tar')))
    return checkpoint


def save_model(model, list_name, client,path):
    np_path = './SaveModel'+path+'/Model_'+str(client)+'/'
    if os.path.isdir(np_path) == False:
        os.makedirs(np_path)

    for name in list_name:
        temp_num = model.state_dict()[name].cpu().numpy()
        np.save(np_path + "%s.ndim" % (name), temp_num)



def create_splits(argument_dict):

    splits = []

    if argument_dict['mode']=='distillation':





        total_len_train=(1-argument_dict['common_dataset_size'])*0.9
        total_len_valid = (1 - argument_dict['common_dataset_size']) * 0.1




        for i in range(argument_dict['clients']):
            splits.append(total_len_train / (argument_dict['clients']))

        for i in range(argument_dict['clients']):
            splits.append(total_len_valid / (argument_dict['clients'] ))





    else:

        total_len_train = (1 - argument_dict['common_dataset_size']) * 0.9
        total_len_valid = (1 - argument_dict['common_dataset_size']) * 0.1


        for i in range(argument_dict['clients']):
            splits.append(total_len_train / argument_dict['clients'])

        for i in range(argument_dict['clients']):
            splits.append(total_len_valid / argument_dict['clients'])

    return splits



def create_save_folder(argument_dict,path):


    datestring = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = './saved_models'+path+'/' + datestring



    os.makedirs(save_path)
    os.mkdir(os.path.join(save_path,'py_files'))

    f = open(save_path + "/arguments.txt", "w")
    f.write(str(argument_dict))
    f.close()

    for f in os.listdir(os.getcwd()):
            if f.endswith('.py'):
                shutil.copy2(os.path.join(os.getcwd(), f), os.path.join(os.path.join(save_path, 'py_files')))

    return save_path



def activations_calculation(output,activation,target,output_avg,activation_avg,tot_output_avg):



    for i in range(output.size()[0]):

        output_avg[target[i]].append(output[i, target[i]].detach().clone())
        tot_output_avg[target[i]].append(output[i,:].detach().clone())
        activation_avg[target[i]].append(activation[i, :, :, :].detach().clone())





    return   activation_avg, output_avg, tot_output_avg



def alpha_calculation(alpha_cap,per_client_labels, per_client_output_avg_np, per_client_grads_avg_np, inverse, case_alpha, alpha_normalization,
                      closed_form_approximation, alpha_opt, alpha_hyperparams, valid_loader_server,
                      model, optimizer, nclients_div,SoTA_comp):


    per_client_grads_avg_ts = torch.from_numpy(per_client_grads_avg_np)
    per_client_output_avg_ts = torch.from_numpy(per_client_output_avg_np)
    per_client_labels_ts = torch.from_numpy(per_client_labels)




    n_clients = per_client_output_avg_ts.size(0)
    n_images = per_client_output_avg_ts.size(1)
    n_classes = per_client_labels_ts.size(2)


    soft_logits = torch.zeros(n_images, n_classes)

    alpha_cap /= math.sqrt(n_classes)
    client_range = range(n_clients)






    if case_alpha=='a':


        #caseA
        for j in range(n_clients):


            base_embed = per_client_output_avg_ts[j, :, :]
            labels_embed = per_client_labels_ts[j, :, :]
            new_range = (num for num in client_range if num not in {j})

            temp_images= torch.zeros(n_clients-1,per_client_output_avg_ts.size(1),per_client_output_avg_ts.size(2))

            count=0
            for r in new_range:
                temp_images[count,:]=per_client_output_avg_ts[r, :, :]
                count+=1
            embed_i=torch.mean(temp_images,dim=0)



            if closed_form_approximation:

                if alpha_normalization=='abs_clip':
                    alpha = torch.clamp(torch.abs(calculate_optimum_alpha(alpha_cap, embed_i, base_embed, per_client_grads_avg_ts[j, :, :])), min=1e-8, max=alpha_cap)
                elif alpha_normalization=='clip':
                    alpha = torch.clamp(calculate_optimum_alpha(alpha_cap, embed_i, base_embed, per_client_grads_avg_ts[j, :, :]),min=1e-8, max=alpha_cap)
                else:
                    alpha= torch.abs(calculate_optimum_alpha(alpha_cap, embed_i, base_embed, per_client_grads_avg_ts[j,:,:]))


            else:


                if alpha_normalization == 'abs_clip':
                    alpha= torch.clamp(torch.abs(generate_alpha( base_embed.size(), alpha_cap)), min=1e-8, max=alpha_cap)
                elif alpha_normalization == 'clip':
                    alpha = torch.clamp(generate_alpha( base_embed.size(), alpha_cap), min=1e-8, max=alpha_cap)
                else:
                    alpha = generate_alpha( base_embed.size(), alpha_cap)


                if alpha_opt:

                    alpha_learning_rate, alpha_learning_iters, alpha_clf_coef, alpha_l2_coef, alpha_grads_div = alpha_hyperparams

                    # model=0
                        # optimizer=0

                    _, probs_sort_idxs = base_embed.sort(descending=True)




                    alpha, model, optimizer = learn_alpha(model.cuda(), base_embed, labels_embed, embed_i, alpha,
                                                          alpha_learning_iters, alpha_learning_rate,
                                                          alpha_clf_coef, alpha_l2_coef, valid_loader_server,
                                                          optimizer, alpha_grads_div,
                                                          SoTA_comp)








            if inverse=='aggregation1':
                soft_logits += (1-alpha) * base_embed
            elif inverse=='aggregation2':
                soft_logits +=  alpha * base_embed
            elif inverse=='aggregation_inverse1':
                soft_logits +=   (1/(1-alpha))*base_embed
            elif inverse=='aggregation_inverse2':
                soft_logits += (1/alpha)*base_embed
            elif inverse=='mix1':
                soft_logits += (1-alpha) * base_embed + alpha *embed_i

            elif inverse=='mix2':

                soft_logits += alpha * base_embed + (1 - alpha) * embed_i


    elif case_alpha=='b':


        #caseB
        for j in range(n_clients-1):

            target_embed=0
            if j==0:
                base_embed = per_client_output_avg_ts[j, :, :]

            labels_embed=per_client_labels_ts[j,:,:]
            embed_i=per_client_output_avg_ts[j+1, :, :]

            if closed_form_approximation:

                if alpha_normalization=='abs_clip':
                    alpha = torch.clamp(torch.abs(calculate_optimum_alpha(alpha_cap, embed_i, base_embed, per_client_grads_avg_ts[j, :, :])), min=1e-8, max=alpha_cap)
                elif alpha_normalization=='clip':
                    alpha = torch.clamp(calculate_optimum_alpha(alpha_cap, embed_i, base_embed, per_client_grads_avg_ts[j, :, :]),min=1e-8, max=alpha_cap)
                else:
                    alpha= torch.abs(calculate_optimum_alpha(alpha_cap, embed_i, base_embed, per_client_grads_avg_ts[j,:,:]))

            else:

                if alpha_normalization == 'abs_clip':
                    alpha = torch.clamp(torch.abs(generate_alpha( base_embed.size(), alpha_cap)), min=1e-8, max=alpha_cap)
                elif alpha_normalization == 'clip':
                    alpha = torch.clamp(generate_alpha( base_embed.size(), alpha_cap), min=1e-8, max=alpha_cap)
                else:
                    alpha = generate_alpha( base_embed.size(), alpha_cap)



                if alpha_opt:

                    alpha_learning_rate,alpha_learning_iters, alpha_clf_coef, alpha_l2_coef, alpha_grads_div =alpha_hyperparams





                    _, probs_sort_idxs = labels_embed.sort(descending=True)









                    alpha, model, optimizer =learn_alpha(model, base_embed, labels_embed, embed_i, alpha, alpha_learning_iters,alpha_learning_rate,
                                       alpha_clf_coef, alpha_l2_coef, valid_loader_server, optimizer,
                                                          alpha_grads_div, SoTA_comp)


                    #

            if inverse=='mix1':
                base_embed = (1-alpha) * base_embed + alpha *embed_i
            elif inverse=='mix2':
                base_embed =  alpha * base_embed + (1-alpha) * embed_i


        soft_logits=base_embed

    if nclients_div:
        if inverse=='mix1' or inverse=='mix2':
            soft_logits_np = soft_logits.detach().numpy() / n_clients
        else:
            soft_logits_np = soft_logits.detach().numpy() / (n_clients*alpha_cap)

    else:
        soft_logits_np=soft_logits.detach().numpy()





    return soft_logits_np, model, optimizer



def contrastive_loss(mu, temperature, output, activation, model_buffer_size, task_activation_prev, round_activation_prev):

    batch_size=activation.size()[0]

    cos = torch.nn.CosineSimilarity(dim=-1)
    criterion = nn.CrossEntropyLoss().cuda()


    nega = cos(activation.view(batch_size, -1), torch.from_numpy(round_activation_prev).view(batch_size, -1).cuda())
    negat = nega.reshape(-1, 1)



    if model_buffer_size > len(task_activation_prev):
        model_buffer_size=len(task_activation_prev)

    for i in range(model_buffer_size):


        posi = cos(activation.view(batch_size, -1), torch.from_numpy(task_activation_prev[-(i+1)]).view(batch_size, -1).cuda())
        logits = torch.cat((posi.reshape(-1, 1), negat), dim=1)


    logits /= temperature
    labels = torch.zeros(output.size(0)).cuda().long()

    loss = mu * criterion(logits.cpu(), labels.cpu())


    return loss


def incremental_learning_mod(model, numclass, mod):

    if mod=='a':

        weight = model.linear.weight.data
        bias = model.linear.bias.data
        in_feature = model.linear.in_features
        out_feature = model.linear.out_features

        model.linear = nn.Linear(in_feature, numclass, bias=True)
        model.linear.weight.data[:out_feature] = weight
        model.linear.bias.data[:out_feature] = bias

    elif mod=='b':

        weight = model.fc.weight.data
        bias = model.fc.bias.data
        in_feature = model.fc.in_features
        out_feature = model.fc.out_features

        model.fc= nn.Linear(in_feature, numclass, bias=True)
        model.fc.weight.data[:out_feature] = weight
        model.fc.bias.data[:out_feature] = bias



    return model

def multi_scale_output_calculation(out, scale):


    out_f= []



    for s in range(scale):
        a = torch.split(out, out.size()[2] // (s+1), dim=2)
        b = [torch.split(x, out.size()[3] // (s+1), dim=3) for x in a]
        c = [y for x in b for y in x]

        out1= sum(arr.size()[2] for arr in c)
        out2 = sum(arr.size()[3] for arr in c)

        out_s = torch.zeros(out.size()[0], out.size()[1], out1 + out2)
        previous_index=0

        for i in range(len(c)):

            step_w = c[i].size()[2]
            step_h = c[i].size()[3]

            out_s1 = 0

            for j in range(c[i].size()[2]):
                out_s1 += c[i][:, :, j, :]/c[i].size()[2]

            out_s2 = 0

            for j in range(c[i].size()[3]):
                out_s2 += c[i][:, :, :, j]/c[i].size()[3]

            out_s3 = torch.cat((out_s1, out_s2), dim=2)


            out_s[:,:,previous_index:previous_index+(step_w+step_h)]=out_s3
            previous_index=previous_index+step_w+step_h
            out_f.append(out_s)


    out_final=torch.cat(out_f, dim=2)






    return out_final