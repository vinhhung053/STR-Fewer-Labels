import os
import sys
import time
import random
import argparse
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import random


from utils import AttnLabelConverter, Averager, adjust_learning_rate
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset, Gradual_Dataset
from model import Model
from test import validation, benchmark_all_eval
from modules.semi_supervised import PseudoLabelLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Adaptation area """

def select_intermediate_domain(opt, model, unlabel_data, converter, flag = True):
    if os.path.isfile(opt.adapted_path):
        with open(opt.adapted_path, 'rb') as f:
            adapted_data = pickle.load(f)           # load adapted data dictionary
    else:
        adapted_data = {}

    if os.path.isfile(opt.remain_path):
        with open(opt.remain_path, 'rb') as f:
            remain_data = pickle.load(f)            # load remain data dictionary
    else:
        remain_data = []

    if (len(remain_data) == 0 and len(adapted_data) != 0):
        return adapted_data, []
    
    if (len(remain_data) == 0 and len(adapted_data) == 0):
        remain_data = list(range(opt.data_size))

    if (not flag):
        with open(opt.adapting_path, 'rb') as f:
                adapting_data = pickle.load(f)
        return adapted_data, adapting_data

    unlabel_loader_remain = unlabel_data.get_dataloader(
        remain_data,
        batch_size=opt.batch_size_unlabel,
        shuffle=False
    )

    pseudo_dict = dict()
    overconfident_data = dict()
    confidence_list = list()
    with torch.no_grad():
        for image_tensors, index_unlabel in unlabel_loader_remain:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            text_for_pred = (
                    torch.LongTensor(batch_size)
                    .fill_(converter.dict["[SOS]"])
                    .to(device)
                )
            preds = model(image, text_for_pred, is_train=False)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_index, pred, pred_max_prob in zip(
                index_unlabel, preds_str, preds_max_prob
            ):
                pred_EOS = pred.find("[EOS]")
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
                
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                # if confidence_score higher than threshold_above, we won't use them to adapt
                if (confidence_score > opt.threshold_above):
                    overconfident_data[img_index] = pred
                else:
                    # save pseudo_label
                    pseudo_dict[img_index] = pred

                    # save confidence_list
                    confidence_list.append([img_index ,confidence_score.item()])

    confidence_bound = np.partition(np.array(confidence_list)[:, 1], -opt.n_samples)[::-1][opt.n_samples-1]
    if float(confidence_bound) < opt.threshold_below:
        confidence_bound = opt.threshold_below      # threshhold

    # select n data having best confidence score
    adapting_data = {index: pseudo_dict[index] for index, conf in confidence_list if conf >= confidence_bound}
    
    adapted_data.update(overconfident_data)
    pseudo_data = adapted_data.copy()
    adapted_data.update(adapting_data)       # add selected data in order to save

    # save adapted file
    with open(opt.adapted_path, "wb") as f:
        pickle.dump(adapted_data, f)
            
    #save adapting file
    with open(opt.adapting_path, 'wb') as f:
        pickle.dump(adapting_data, f)

    # save remain file        
    remain_data = np.setdiff1d(remain_data, list(adapting_data.keys()))
    with open(opt.remain_path, 'wb') as f:
        pickle.dump(remain_data, f)

    # random_number = random.randint(0, 100)
    # print(adapting_data[list(adapting_data.keys())[random_number]])
    # unlabel_data._dataset[list(adapting_data.keys())[random_number]][0].show()
    # time.sleep(5)

    return pseudo_data, adapting_data


def train(opt, log):

    """dataset preparation"""

    select_data = ["MJ", "ST"]          # source domain

    # set batch_ratio for each data.
    batch_ratio = [round(1 / len(select_data), 3)] * len(select_data)

    train_loader = Batch_Balanced_Dataset(
        opt, opt.train_data, select_data, batch_ratio, log
    )

    # validation data
    AlignCollate_valid = AlignCollate(opt, mode="test")
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt, mode="test"
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size_unlabel,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=False,
    )

    # adaptation data
    # select_data_unlabel = ['U2.TextVQA', "U3.STVQA"]
    select_data_unlabel = ["U1.Book32", "U2.TextVQA", "U3.STVQA"]
    unlabel_data = Gradual_Dataset(opt, opt.dataset_root_unlabel, select_data_unlabel, log)

    print("-" * 80)

    """ model configuration """

    converter = AttnLabelConverter(opt.character)
    opt.sos_token_index = converter.dict["[SOS]"]
    opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)

    if (opt.checkpoint_path != ""):
        checkpoint = torch.load(opt.checkpoint_path)
    else:
        checkpoint = {
            'iteration': 0,
            'best_score': -1,
            'model': torch.load(opt.saved_model, map_location=device),
        }

    model = Model(opt)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    fine_tuning_log = f"### loading pretrained model from {opt.saved_model}\n"

    model.load_state_dict(checkpoint['model'])

    """ setup loss """

    # ignore [PAD] token
    criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict["[PAD]"]).to(
        device
    )

    # criterion_SemiSL = PseudoLabelLoss(opt, converter, criterion)

    # loss averager
    train_loss_avg = Averager()
    semi_loss_avg = Averager()  # semi supervised loss avg

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    """ final options """
    # print(opt)
    opt_log = "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            opt_log += f"{str(k)}: {str(v)}\n"
    opt_log += "---------------------------------------\n"
    # print(opt_log)
    log.close()

    start_time = time.time()
    best_score = checkpoint['best_score']
    
    print("Start training...")

    """ start training """
    start_iter = checkpoint['iteration']

    """ select intermediate domain """

    print("-" * 80)
    print("select intermediate domain")

    if (start_iter % opt.num_iter_GDA != 0):

        # setup optimizer
        optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])

        if "super" in opt.schedule:
            cycle_momentum = False

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt.lr,
                cycle_momentum=cycle_momentum,
                div_factor=20,
                final_div_factor=1000,
                total_steps=opt.num_iter,
            )
            scheduler.load_state_dict(checkpoint['scheduler'])
    
        adapted_dict, adapting_dict = select_intermediate_domain(
            opt, model, unlabel_data, converter, flag = False)

        unlabel_loader_adapted = unlabel_data.get_dataloader(
            list(adapted_dict.keys()),
            batch_size=opt.batch_size,
            shuffle=True
        )

        unlabel_loader_adapting = unlabel_data.get_dataloader(
            list(adapting_dict.keys()),
            batch_size=opt.batch_size,
            shuffle=True
        )

    # training loop
    for iteration in tqdm(
        range(start_iter + 1, opt.num_iter + 1),
        total=opt.num_iter - start_iter,
        position=0,
        leave=True,
    ):
        if (iteration - 1) % opt.num_iter_GDA == 0:
            adapted_dict, adapting_dict = select_intermediate_domain(
                opt, model, unlabel_data, converter, flag = True)
            
            if len(adapting_dict) < 300:
                print('Early Stopping')
                break

            # setup optimizer
            optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr)

            if "super" in opt.schedule:
                cycle_momentum = False

                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt.lr,
                    cycle_momentum=cycle_momentum,
                    div_factor=20,
                    final_div_factor=1000,
                    total_steps=opt.num_iter,
                )
            
            unlabel_loader_adapted = unlabel_data.get_dataloader(
                list(adapted_dict.keys()),
                batch_size=opt.batch_size,
                shuffle=True
            )

            unlabel_loader_adapting = unlabel_data.get_dataloader(
                list(adapting_dict.keys()),
                batch_size=opt.batch_size,
                shuffle=True
            )

        # validation part.
        # To see training progress, we also conduct validation when 'iteration == 1'
        if (iteration - 1) % opt.val_interval == 0:

            # for validation log
            with open(f"./saved_models/{opt.exp_name}/log_train.txt", "a") as log:
                model.eval()
                with torch.no_grad():
                    (
                        valid_loss,
                        current_score,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data,
                    ) = validation(model, criterion, valid_loader, converter, opt)
                model.train()

                # keep best score (accuracy or norm ED) model on valid dataset
                # Do not use this on test datasets. It would be an unfair comparison
                # (training should be done without referring test set).
                if current_score > best_score:
                    best_score = current_score
                    torch.save(
                        model.state_dict(),
                        f"./saved_models/{opt.exp_name}/best_score.pth",
                    )

                # save checkpoint
                checkpoint = {
                    'iteration': iteration - 1,
                    'best_score': best_score,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }

                checkpoint_path = f'./checkpoint/{opt.exp_name}_checkpoint.pt'
                torch.save(checkpoint, checkpoint_path)

                # validation log: loss, lr, score (accuracy or norm ED), time.
                lr = optimizer.param_groups[0]["lr"]
                elapsed_time = time.time() - start_time
                valid_log = f"\n[{iteration}/{opt.num_iter}] Train_loss: {train_loss_avg.val():0.5f}, Valid_loss: {valid_loss:0.5f}"
                valid_log += f", Semi_loss: {semi_loss_avg.val():0.5f}\n"
                valid_log += f'{"Current_score":17s}: {current_score:0.2f}, Current_lr: {lr:0.7f}\n'
                valid_log += f'{"Best_score":17s}: {best_score:0.2f}, Infer_time: {infer_time:0.1f}, Elapsed_time: {elapsed_time:0.1f}'

                # show some predicted results
                dashed_line = "-" * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
                for gt, pred, confidence in zip(
                    labels[:20], preds[:20], confidence_score[:20]
                ):

                    gt = gt[: gt.find("[EOS]")]
                    pred = pred[: pred.find("[EOS]")]

                    predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += f"{dashed_line}"
                valid_log = f"{valid_log}\n{predicted_result_log}"
                print(valid_log)
                # log.write(valid_log + "\n")

                opt.writer.add_scalar(
                    "train/train_loss", float(f"{train_loss_avg.val():0.5f}"), iteration
                )
                opt.writer.add_scalar(
                    "train/semi_loss", float(f"{semi_loss_avg.val():0.5f}"), iteration
                )
                opt.writer.add_scalar("train/lr", float(f"{lr:0.7f}"), iteration)
                opt.writer.add_scalar(
                    "train/elapsed_time", float(f"{elapsed_time:0.1f}"), iteration
                )
                opt.writer.add_scalar(
                    "valid/valid_loss", float(f"{valid_loss:0.5f}"), iteration
                )
                opt.writer.add_scalar(
                    "valid/current_score", float(f"{current_score:0.2f}"), iteration
                )
                opt.writer.add_scalar(
                    "valid/best_score", float(f"{best_score:0.2f}"), iteration
                )

                train_loss_avg.reset()
                semi_loss_avg.reset()

        """ loss of source domain """
        image_tensors, labels = train_loader.get_batch()

        image = image_tensors.to(device)
        labels_index, _ = converter.encode(
            labels, batch_max_length=opt.batch_max_length
        )

        # default recognition loss part
        preds = model(image, labels_index[:, :-1])  # align with Attention.forward
        target = labels_index[:, 1:]  # without [SOS] Symbol
        loss_source = criterion(
            preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
        )

        """ loss of adapted domain """
        images_adapted, indexs_adapted = next(iter(unlabel_loader_adapted))
        images_adapted = images_adapted.to(device)          
        labels_adapted = [adapted_dict[index] for index in indexs_adapted]
        labels_adpated_index, _ = converter.encode(
            labels_adapted, batch_max_length=opt.batch_max_length
        )

        preds_adapted = model(images_adapted, labels_adpated_index[:, :-1])  # align with Attention.forward
        target_adapted = labels_adpated_index[:, 1:]  # without [SOS] Symbol
        loss_adapted = criterion(
            preds_adapted.view(-1, preds_adapted.shape[-1]), target_adapted.contiguous().view(-1)
        )

        # """ loss of semi """
        # # semi supervised part (SemiSL)
        # images_unlabel, indexs_unlabel = next(iter(unlabel_loader_adapting))
        # images_unlabel = images_unlabel.to(device)
        # loss_SemiSL = criterion_SemiSL(images_unlabel, model)

        """ loss of semi """
        # semi supervised part (SemiSL)
        images_unlabel, indexs_unlabel = next(iter(unlabel_loader_adapting))
        images_unlabel = images_unlabel.to(device)
        labels_adapting = [adapting_dict[index] for index in indexs_unlabel]
        labels_adpating_index, _ = converter.encode(
            labels_adapting, batch_max_length=opt.batch_max_length
        )

        preds_adapting = model(images_unlabel, labels_adpating_index[:, :-1])  # align with Attention.forward
        target_adapting = labels_adpating_index[:, 1:]  # without [SOS] Symbol
        loss_SemiSL = criterion(
            preds_adapting.view(-1, preds_adapting.shape[-1]), target_adapting.contiguous().view(-1)
        )

        loss = loss_source + loss_adapted + loss_SemiSL
        # loss = loss_source + loss_SemiSL
        # semi_loss_avg.add(loss_SemiSL)

        model.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), opt.grad_clip
        )  # gradient clipping with 5 (Default)
        optimizer.step()
        # train_loss_avg.add(loss)

        if "super" in opt.schedule:
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, iteration, opt)

    """ Evaluation at the end of training """
    print("Start evaluation on benchmark testset")
    """ keep evaluation model and result logs """
    os.makedirs(f"./result/{opt.exp_name}", exist_ok=True)
    os.makedirs(f"./evaluation_log", exist_ok=True)
    saved_best_model = f"./saved_models/{opt.exp_name}/best_score.pth"
    # os.system(f'cp {saved_best_model} ./result/{opt.exp_name}/')
    model.load_state_dict(torch.load(f"{saved_best_model}"))

    opt.eval_type = "benchmark"
    model.eval()
    with torch.no_grad():
        total_accuracy, eval_data_list, accuracy_list = benchmark_all_eval(
            model, criterion, converter, opt
        )

    opt.writer.add_scalar(
        "test/total_accuracy", float(f"{total_accuracy:0.2f}"), iteration
    )
    for eval_data, accuracy in zip(eval_data_list, accuracy_list):
        accuracy = float(accuracy)
        opt.writer.add_scalar(f"test/{eval_data}", float(f"{accuracy:0.2f}"), iteration)

    print(
        f'finished the experiment: {opt.exp_name}, "CUDA_VISIBLE_DEVICES" was {opt.CUDA_VISIBLE_DEVICES}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="data_CVPR2021/training/label/",
        help="path to training dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="data_CVPR2021/validation/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="number of data loading workers"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
    parser.add_argument(
        "--num_iter", type=int, default=1000, help="number of iterations to train for"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=100,
        help="Interval between each validation",
    )
    parser.add_argument(
        "--log_multiple_test", action="store_true", help="log_multiple_test"
    )
    parser.add_argument(
        "--FT", type=str, default="init", help="whether to do fine-tuning |init|freeze|"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    """ Optimizer """
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer |sgd|adadelta|adam|"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="learning rate, default=1.0 for Adadelta, 0.0005 for Adam",
    )
    parser.add_argument(
        "--sgd_momentum", default=0.9, type=float, help="momentum for SGD"
    )
    parser.add_argument(
        "--sgd_weight_decay", default=0.000001, type=float, help="weight decay for SGD"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.95,
        help="decay rate rho for Adadelta. default=0.95",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8"
    )
    parser.add_argument(
        "--schedule",
        default="super",
        nargs="*",
        help="(learning rate schedule. default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER",
    )
    parser.add_argument(
        "--lr_drop_rate",
        type=float,
        default=0.1,
        help="lr_drop_rate. default is the same setting with ASTER",
    )
    """ Model Architecture """
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    """ Data processing """
    parser.add_argument(
        "--total_data_usage_ratio",
        type=str,
        default="1.0",
        help="total data usage ratio, this ratio is multiplied to total number of data.",
    )
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        help="character label",
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
    )
    parser.add_argument(
        "--Aug",
        type=str,
        default="None",
        help="whether to use augmentation |None|Blur|Crop|Rot|",
    )
    """ Semi-supervised learning """
    parser.add_argument(
        "--semi",
        type=str,
        default="Pseudo",
        help="whether to use semi-supervised learning |None|PL|MT|",
    )
    parser.add_argument(
        "--MT_C", type=float, default=1, help="Mean Teacher consistency weight"
    )
    parser.add_argument(
        "--MT_alpha", type=float, default=0.999, help="Mean Teacher EMA decay"
    )
    parser.add_argument(
        "--model_for_PseudoLabel", default="TRBA-Baseline-synth.pth", help="trained model for PseudoLabel"
    )
    parser.add_argument(
        "--self_pre",
        type=str,
        default="RotNet",
        help="whether to use `RotNet` or `MoCo` pretrained model.",
    )
    """ exp_name and etc """
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )
    parser.add_argument(
        "--saved_model", default="TRBA-Baseline-synth.pth", help="path to model to continue training"
    )
    """ adaptation """
    parser.add_argument(
        "--dataset_root_unlabel", 
        default="data_CVPR2021/training/unlabel/", 
        help="path to unlabel data"
    )
    parser.add_argument(
        "--adapted_path", 
        default="adaptation/adapted.pkl", 
        help="path to adapted data dictionary"
    )
    parser.add_argument(
        "--adapting_path", 
        default="adaptation/adapting.pkl", 
        help="path to adapting data dictionary"
    )
    parser.add_argument(
        "--remain_path", 
        default="adaptation/remain.pkl", 
        help="path to remain data list"
    )
    parser.add_argument(
        "--checkpoint_path", 
        default="", 
        help="checkpoint"
    )
    parser.add_argument("--batch_size_unlabel", type=int, default=2048, help="input batch size (unlabel data)")
    parser.add_argument("--data_size", type=int, default=2048000, help="total number of intermediate data")
    parser.add_argument("--threshold_above", type=float, default=0.7, help="threshold of confidence score")
    parser.add_argument("--threshold_below", type=float, default=0.3, help="threshold of confidence score")
    parser.add_argument("--n_samples", type=int, default=20480, help="n samples of each adapt step")
    parser.add_argument("--num_iter_GDA", type=int, default=300, help="number of iterations to train for each adaptation step")


    opt = parser.parse_args()

    # model: TRBA
    opt.Transformation = "TPS"
    opt.FeatureExtraction = "ResNet"
    opt.SequenceModeling = "BiLSTM"
    opt.Prediction = "Attn"

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print(
            "We recommend to use 1 GPU, check your GPU number, you would miss CUDA_VISIBLE_DEVICES=0 or typo"
        )
        print("To use multi-gpu setting, remove or comment out these lines")
        sys.exit()

    if sys.platform == "win32":
        opt.workers = 0

    """ directory and log setting """
    if not opt.exp_name:
        opt.exp_name = f"Seed{opt.manual_seed}-{opt.model_name}"

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)
    log = open(f"./saved_models/{opt.exp_name}/log_train.txt", "a")
    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )
    log.write(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}\n"
    )
    os.makedirs(f"./tensorboard", exist_ok=True)
    opt.writer = SummaryWriter(log_dir=f"./tensorboard/{opt.exp_name}")

    train(opt, log)
