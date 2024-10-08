import argparse
import datetime

def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default = None, help='dataset path')
    parser.add_argument('--pretrained', default = None, help='pretrained model path')

    parser.add_argument('--samplings', type = int, default = 4, help ='sampling images')
    parser.add_argument('--num_classes', type = int, default = 3, help ='Number of classes')
    parser.add_argument('--channels', type = int, default = 1, help = 'Number of image channles')
    parser.add_argument('--resume', default = False, help = 'resume training')
    parser.add_argument('--resume_epoch', default = 0, help = 'epoch of resume step')
    parser.add_argument('--model', default = None, help = 'model name')
    parser.add_argument('--alpha', type = float, default = 10, help = 'cycle loss weight')
    parser.add_argument('--logit_alpha', type = float, default = 0.1, help = 'CR loss weight')
    parser.add_argument('--log_file', type = str, default = 'log.csv',
            help='log file to keep track losses of each epoch')

    # Additional options for training
    parser.add_argument('--epochs', type = int, default = 70, help = 'number of epochs')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--state', default = 'train', help = 'state')
    parser.add_argument('--image_size', type = int, default = 128, help = 'image size')
    parser.add_argument('--objective', default = 'ddpm', help = 'objective')
    parser.add_argument('--schedule', default = 'linear', help = 'schedule')
    parser.add_argument('--device', default = "cuda", help = 'device')
    parser.add_argument('--lr', type = float, default = 2e-4, help = 'learning rate')
    parser.add_argument('--train', action = 'store_true', help = 'train flag')
    parser.add_argument('--margin', type = float, default = 0.1, help = 'margin for contrastive learning')
    parser.add_argument('--save_term', type = int, default = 1, help = 'margin for contrastive learning')
    today =  datetime.datetime.now().strftime("%Y%m%d")
    parser.add_argument('--folder_name', default = today, help='pretrained model path')

    
    # Optional augmentation
    parser.add_argument('--num_synthetic', type = int, default = 1000, help = 'margin for contrastive learning')
    return parser.parse_args()