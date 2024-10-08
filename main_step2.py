import os
import torch
import torch.nn as nn
import tqdm
from torch import optim
from utils.utils_step2 import *
from utils.utils import *
from module.modules import *
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from diffusion.diffusion import Diffusion
from options.base_options import get_default_args
from copy import deepcopy

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    args.run_name = "diffusion_{}_{}_{}".format(args.objective, args.schedule, args.folder_name)
    setup_logging(args.run_name)
    print(args.run_name)

    model = Cont_UViT(img_size=128, 
        patch_size=8, 
        in_chans=args.channels, 
        embed_dim=512, 
        depth=12, 
        num_heads=8, 
        num_classes=args.num_classes,
        mlp_ratio=4, 
        qkv_bias=False, 
        mlp_time_embed=False).to(args.device)

    model_class= D_Cube(img_size=128, 
        patch_size=8, 
        in_chans=args.channels, 
        embed_dim=512, 
        depth=12, 
        num_heads=8, 
        num_classes=args.num_classes,
        mlp_ratio=4, 
        qkv_bias=False, 
        mlp_time_embed=False).to('cuda')
    
    optimizer_class = optim.AdamW(model_class.parameters(), lr=args.lr,weight_decay=0.03,betas=(0.9, 0.999))
    model = torch.nn.DataParallel(model)
    model_class = torch.nn.DataParallel(model_class)


    if args.pretrained is not None:
        checkpoint_model = torch.load(args.pretrained)
        model.load_state_dict(checkpoint_model)
        print(f"weight sharing with pretrained model")

    dataloader = get_data_step2(args.dataset_path, state='train', batch_size = args.batch_size, size = args.image_size)
    dataloader_test= get_data_step2(args.dataset_path, state='test', batch_size = args.batch_size, size = args.image_size)    
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    diffusion = Diffusion(img_size=args.image_size, device=args.device, objective=args.objective)
    mse = nn.MSELoss()

    logger = SummaryWriter(os.path.join("logs", args.run_name))
    l = len(dataloader)
    log = os.path.join("results", args.run_name, args.log_file)
    os.makedirs(os.path.dirname(log), exist_ok=True)

    with open(log, mode='w') as f:
                f.write("epoch, train_loss, acc, recall, prcision, f1_score, benign, malign, normal\n")

    model_conditional_state_dict = checkpoint_model
    model_class_dict = model_class.state_dict()

    for name, param in checkpoint_model.items():
        if 'time_embed' in name or 'label_emb' in name or 'pos_embed' in name or 'patch_embed' in name or 'in_blocks' in name or \
            'mid_block' in name or 'out_blocks' in name:
            model_class_dict[name] =  model_conditional_state_dict[name]

    model_class.load_state_dict(model_class_dict)

    layers_to_unfreeze = ['conv_gray', 'conv_channel', 
                          'conv4resnet','sub_features','fc_layer','class_layer1','class_layer2','class_layer3','class_layer4',
                          'class_layer5','class_layer6','norm1_D','norm2_D','norm3_D', 'norm_f']
    for name, param in model_class.named_parameters():
        
        if any(layer_name in name for layer_name in layers_to_unfreeze):
            param.requires_grad = True
        else:
            param.requires_grad = False

    
    for i in range(args.epochs):
        epoch = i + 1
        logging.info(f"Starting epoch {epoch}:")
            
        acc_avg=0.0
        num_batch=0.0
        
        model_class.train()
        model.eval()
        
        pbar_new = tqdm.tqdm(dataloader)
        for i, image in enumerate(pbar_new):
            images, label, ori_image, images_flip, ori_image_flip = image
            images = images.float().to(args.device)
            ori_image = ori_image.float().to(args.device)
            images_flip = images_flip.float().to(args.device)
            ori_image_flip = ori_image_flip.float().to(args.device)
            label = label.long().to(args.device)
            n_b = label.shape[0]


            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            output = model_class(images, t, ori_image)
            output_flip = model_class(images_flip, t, ori_image_flip)

            label_predict_index = torch.argmax(output, dim=-1)
            ### CR Loss
            logit_loss = mse(output,output_flip)

            with torch.no_grad():
                t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
                x_t, noise = diffusion.noise_images(images, t)
                predicted_label_noise, _, _, _ = model(x_t, x_t, t, label_predict_index, label_predict_index)
                gt_label_noise, _, _, _ = model(x_t, x_t, t, label, label)
            
            ### Cylce Loss
            noise_loss = mse(gt_label_noise, predicted_label_noise)
            ### CE Loss
            label_loss = criterion(output, label)
            loss_final = args.alpha * noise_loss + label_loss + args.logit_alpha*logit_loss

            optimizer_class.zero_grad()
            loss_final.backward()
            optimizer_class.step()

            if epoch == 1:
                model_ema=deepcopy(model_class)
            if epoch !=1: 
                moving_average(model_ema, model_class, 0.999)


            pbar_new.set_postfix(Loss = loss_final.item())
            logger.add_scalar("Loss", loss_final.item(), global_step = epoch * l + i)

            acc = torch.sum(torch.eq(label_predict_index, label)).item()
            acc_avg += acc 
            num_batch += n_b

        correct_counts = [0] * args.num_classes
        total_counts = [0] * args.num_classes

        acc_avg_t=0.0
        num_batch_t=0.0
        itr = 0
        recall_t = np.zeros(args.num_classes)
        precision_t = np.zeros(args.num_classes)
        f1_t = np.zeros(args.num_classes)
        all_true_labels = []
        all_predicted_labels = []
        best_value = None
        model_class.eval()

        with torch.no_grad():
            for i, image in enumerate(tqdm.tqdm(dataloader_test)):
                images, label, ori_image, _, _ = image
                images = images.float().to(args.device)
                ori_image = ori_image.float().to(args.device)
                label = label.long().to(args.device)
                n_b = label.shape[0]

                t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
                output = model_class(images, t, ori_image)
                label_predict_index = torch.argmax(output, dim=-1)
                label_predict_index = label_predict_index.cpu().detach().numpy()
                label = label.cpu().detach().numpy()

                acc = np.sum(np.equal(label_predict_index,label))
                all_true_labels.extend(label)
                all_predicted_labels.extend(label_predict_index)

                for lb in range(args.num_classes):
                    lb_indices = (label == lb)
                    correct_counts[lb] += np.sum(label_predict_index[lb_indices] == lb)
                    total_counts[lb] += np.sum(lb_indices)

                itr += 1
                acc_avg_t += acc 
                num_batch_t += n_b

            recall_t = recall_score(all_true_labels, all_predicted_labels, average=None, zero_division=1)* 100
            precision_t = precision_score(all_true_labels, all_predicted_labels, average=None, zero_division=1)* 100
            f1_t = f1_score(all_true_labels, all_predicted_labels, average=None, zero_division=1)* 100
            accu = (acc_avg_t/num_batch_t) * 100

            print("###ACCURACY###")
            for lb in range(args.num_classes):
                lbl_acc = correct_counts[lb] / total_counts[lb] if total_counts[lb] != 0 else 0
                print(f'- Accuracy for label {lb}: {lbl_acc * 100:.2f}% ({correct_counts[lb]} / {total_counts[lb]})')
            print('test classification accuracy:{:4f}%'.format(accu))

            print("###RECALL###")           
            for i, recall in enumerate(recall_t):
                    print(f"- recall of class {i}: {recall:.2f}%")
            avg_recall = np.mean(recall_t)
            print('test classification recall:{:4f}%'.format(avg_recall))

            print("###PRECISION###")
            for i, recall in enumerate(precision_t):
                    print(f"- precision of class {i}: {recall:.2f}%")
            avg_precision = np.mean(precision_t)

            print('test classification precision:{:4f}%'.format(avg_precision))
            print("###F1###")
            for i, recall in enumerate(f1_t):
                    print(f"- f1 of class {i}: {recall:.2f}%")
            avg_f1 = np.mean(f1_t)
            print('test classification f1:{:4f}%'.format(avg_f1))

            combined_metric = accu / 100 + np.mean(recall_t) + np.mean(precision_t) + np.mean(f1_t)

            with open(log, mode='a') as f:
                f.write("{},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{},{},{}\n".format(
                    epoch, loss_final.item(), accu, avg_recall, avg_precision, avg_f1, correct_counts[0],correct_counts[1],correct_counts[2]
                ))

            if best_value is None or combined_metric > best_value:
                best_value = combined_metric
                torch.save(model_class.state_dict(), os.path.join("models", args.run_name, f"diffuclassifier_best_model.pt"))
                print(f"best value : acc {accu:.3f} at epoch {epoch}")
                print( accu, avg_recall, avg_precision, avg_f1)

            if epoch%10 == 0 :
                 torch.save(model_class.state_dict(), os.path.join("models", args.run_name, f"diffuclassifier_{epoch}.pt"))

        if epoch % args.save_term == 0:
            sampled_images, classes = diffusion.sample(model, n= args.samplings, num_classes = args.num_classes, channels = args.channels)
            save_images(sampled_images, os.path.join("results", args.run_name, f"sample_images_{epoch}.png"))
            print("######### model save #########")

if __name__ == '__main__':

    args = get_default_args()
    train(args)
