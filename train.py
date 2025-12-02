import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# CUDA_VISIBLE_DEVICES=0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm.auto import tqdm
import json
import time
import shutil
import copy

import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import random
from src.utils import get_seg_color, copy_files_to_output_dir, IOStream
from src.draw import TrainingMetricsPlotter
# from dataset.PartnetEpc import PartnetEpc
from dataset.PartnetEpcAllG import PartnetEpc
from dataset.PartnetEpcAllGNoise import PartnetEpcNoise
from model.SegmentorNewAllG import SegmentorNew

dataset_dict = {"PartnetE": PartnetEpc, "PartnetE_noise": PartnetEpcNoise}
# from torch.utils.tensorboard import SummaryWriter
# all_categories = ['Bottle', 'Box', 'Bucket', 'Camera', 'Cart', 'Chair', 'Clock', 'CoffeeMachine', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Laptop', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Phone', 'Pliers', 'Printer', 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
category_split = {
"c1" : ['Camera', 'Laptop', 'Safe', 'Phone', 'CoffeeMachine', 'Cart', 'Refrigerator', 'Box', 'Bucket', 'Clock', 'Dishwasher', 'Dispenser', 'Display'],
"c2" : ['Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Oven', 'Pen'],
"c3" : ['Pliers', 'Printer', 'Remote', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window']
}
all_categories = ['Bottle', 'Phone','CoffeeMachine','Laptop',  'Cart', 'Refrigerator',  'Box', 'Bucket', 'Camera', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
# all_categories = ['CoffeeMachine','Laptop',  'Bottle',  'Cart', 'Refrigerator',  'Box', 'Bucket', 'Camera', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
# all_categories = ["Mouse", 'Bottle', "Camera", "KitchenPot" ,"Laptop" ,"Safe", 'Phone','CoffeeMachine', 'Cart', 'Refrigerator',  'Box', 'Bucket', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote','Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
def check(view, img, label, pred, sam_masks_label, num_label):
    img = img.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    sam_masks_label = sam_masks_label.detach().cpu().numpy()
    
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f"output/check/output_{view}_image.png", img)
    
    label = label.reshape(-1)
    rgb = get_seg_color(label, num_label)
    rgb = rgb.reshape(800,800,3)
    rgb  = (rgb  * 255).astype(np.uint8) 
    cv2.imwrite(f"output/check/output_{view}_label.png", rgb)
    
    pred = pred.reshape(-1)
    pred = pred+1
    pred[label==0]=0
    rgb = get_seg_color(pred, num_label)
    rgb[label==0]=0
    rgb = rgb.reshape(800,800,3)
    rgb  = (rgb  * 255).astype(np.uint8) 
    cv2.imwrite(f"output/check/output_{view}_pred.png", rgb)
    
    sam_masks_label = sam_masks_label.reshape(-1)
    sam_masks_label = sam_masks_label+1
    sam_masks_label[label==0]=0
    rgb = get_seg_color(sam_masks_label, sam_masks_label.max()+1)
    rgb[label==0]=0
    rgb = rgb.reshape(800,800,3)
    rgb  = (rgb  * 255).astype(np.uint8) 
    cv2.imwrite(f"output/check/output_{view}_mask.png", rgb)
    pass

def calc_miou(pred, target, num_classes):
    
    miou = 0
    cnt = 0
    for i in range(1, num_classes):
    # for i in range(0,1):
        pred_ind = (pred==i)
        target_ind = (target==i)
        if target_ind.sum()!=0:
            I = pred_ind&target_ind
            I = I.sum()
            U = pred_ind|target_ind
            U = U.sum()
            miou += I/U
            cnt+=1
    if cnt==0:
        return -1
    miou /= cnt
    return miou.cpu().data

def sam_mask_vote(predicted,sam_masks_label):
    for b in range(predicted.shape[0]):   
        for i in range(sam_masks_label[b].max()+1):
            ind = (sam_masks_label[b]==i).nonzero(as_tuple=True)
            if len(ind[0])==0:
                continue
            pred_mask_i = predicted[b][ind]
            unique_elements, counts = torch.unique(pred_mask_i, return_counts=True)
            predicted[b][ind]=unique_elements[torch.argmax(counts)]
    return predicted
      
def check(pc_idx, pc_label, predicted, img_ori, mask_label, num_label):
    view_num = pc_idx.shape[0]
    invalid_ind = pc_idx==-1
    img_label = pc_label[pc_idx]+1
    img_label[invalid_ind]=0
    img_pred = predicted[pc_idx]+1
    img_pred[invalid_ind]=0
    for view in range(view_num):
        check(0, img_ori[view], img_label[view], img_pred[view], mask_label[view],num_label+1)
        print("good")      

def run_epoch(category, epoch, model, dataloader, num_label, args, io, writer, optimizer=None, mode="train", device=None):
    is_training = (mode == "train" or mode=="self")
    if is_training:
        model.train()
    else:
        model.eval()

    loss_epoch, miou_list = [], []
    
    # if is_training:
    #     accumulation_steps = 8  # 模拟 batch_size = 8
    #     optimizer.zero_grad()   # 初始
    
    losses_list = []
    for idx, batch in enumerate(tqdm(dataloader, desc=mode)):
        # torch.cuda.empty_cache()
        pc_id, pc, pc_label, unseen_pc, unseen_pc_label, img, mask_label, pc_idx, coords, graph, pc_norm = batch
        # print(">>>>>>>> ", pc_id)
        # 将数据转为 Tensor 并移动到指定设备
        pc_id = pc_id.item()
        pc = pc.squeeze(dim=0).to(torch.float32).to(device)
        pc_label = pc_label.squeeze(dim=0).to(torch.long).to(device)
        unseen_pc = unseen_pc.squeeze(dim=0).to(torch.float32).to(device)
        unseen_pc_label = unseen_pc_label.squeeze(dim=0).to(torch.long).to(device)
        img = img.squeeze(dim=0).to(torch.float32).to(device)  # HWC -> CHW
        mask_label = mask_label.squeeze(dim=0).to(torch.long).to(device)
        pc_idx = pc_idx.squeeze(dim=0).to(torch.long).to(device)
        coords = coords.squeeze(dim=0).to(torch.float32).to(device)
        # graph = graph.squeeze(dim=0).to(torch.float32).to(device)
        if args.use_ball_propagate:
            pc_norm = pc_norm.squeeze(dim=0).to(torch.long).to(device)
        img_ori = img
        
        
        if args.pretrain:
            input_pc_id = -1
        if mode == "train":
            input_pc_id = pc_id
        elif mode == "val":
            input_pc_id = pc_id + 10
        elif mode == "test" or mode == "self":
            input_pc_id = -1
        
        logits, loss, n_pc_label, losses = model(input_pc_id, pc, pc_label, img, mask_label, pc_idx, coords, graph, pc_norm, args=args, epoch=epoch, mode=mode)

        predicted = logits.argmax(dim=1)
                        
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            
            # (loss / accumulation_steps).backward()  # ❗️分摊loss，累积梯度
            # loss_epoch.append(loss.item())

            # # 梯度累计步骤
            # if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(dataloader):
            #     optimizer.step()
            #     optimizer.zero_grad()
            
        miou = calc_miou(predicted, n_pc_label, num_label)
        if miou!=-1:
            miou_list.append(miou)

    miou_mean = np.mean(miou_list)
    if is_training:
        mean_loss = np.mean(loss_epoch)
        if epoch == 19:
            io.cprint(f"Final {mode.capitalize()} mIoU {category} : {miou_mean:.4f}, Loss: {mean_loss:.4f}")
        else:
            io.cprint(f"{mode.capitalize()} mIoU {category} : {miou_mean:.4f}, Loss: {mean_loss:.4f}")
        return miou_mean, mean_loss
    else:
        io.cprint(f"{mode.capitalize()} mIoU {category} : {miou_mean:.4f}")
        return miou_mean
    
def run_epoch_self(category, epoch, model, dataloader, num_label, args, io, writer, optimizer=None, mode="self", device=None):
    model.train()

    loss_epoch, miou_list = [], []

    accumulation_steps = 8  # 模拟 batch_size = 8
    optimizer.zero_grad()   # 初始化

    for idx, batch in enumerate(tqdm(dataloader, desc=mode)):
        pc_id, pc, pc_label, unseen_pc, unseen_pc_label, img, mask_label, pc_idx, coords, graph, pc_norm = batch
        pc_id = pc_id.item()
        pc = pc.squeeze(dim=0).to(torch.float32).to(device)
        pc_label = pc_label.squeeze(dim=0).to(torch.long).to(device)
        unseen_pc = unseen_pc.squeeze(dim=0).to(torch.float32).to(device)
        unseen_pc_label = unseen_pc_label.squeeze(dim=0).to(torch.long).to(device)
        img = img.squeeze(dim=0).to(torch.float32).to(device)
        mask_label = mask_label.squeeze(dim=0).to(torch.long).to(device)
        pc_idx = pc_idx.squeeze(dim=0).to(torch.long).to(device)
        coords = coords.squeeze(dim=0).to(torch.float32).to(device)
        if args.use_ball_propagate:
            pc_norm = pc_norm.squeeze(dim=0).to(torch.long).to(device)
        img_ori = img
        
        input_pc_id = -1

        logits, loss, n_pc_label, losses = model(
            input_pc_id, pc, pc_label, img, mask_label, pc_idx, coords, graph, pc_norm,
            args=args, epoch=epoch, mode=mode
        )

        predicted = logits.argmax(dim=1)
        
        (loss / accumulation_steps).backward()  # ❗️分摊loss，累积梯度
        loss_epoch.append(loss.item())

        # 梯度累计步骤
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        miou = calc_miou(predicted, n_pc_label, num_label)
        if miou != -1:
            miou_list.append(miou)

    miou_mean = np.mean(miou_list)
    mean_loss = np.mean(loss_epoch)
    if epoch == 19:
        io.cprint(f"Final {mode.capitalize()} mIoU {category} : {miou_mean:.4f}, Loss: {mean_loss:.4f}")
    else:
        io.cprint(f"{mode.capitalize()} mIoU {category} : {miou_mean:.4f}, Loss: {mean_loss:.4f}")
    return miou_mean, mean_loss

def train(args, category, io, writer, dataload_train,dataload_val,dataload_test, seed=0):
    bs = args.batch_size
    lr = args.lr
    partnete_meta = json.load(open("PartNetE_meta.json"))
    num_label = len(partnete_meta[category]) + 1  
    best_model_path = os.path.join(args.output_dir, f"{category}_best_model_{seed}.pth")
    model = SegmentorNew(num_labels=num_label, args=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
        
    if args.use_ave_best:
        ave_eli_dir = "res2/ave_eli"
        loaded_params = torch.load(os.path.join(ave_eli_dir, f"{category}_best_model.pth"))
        model.load_state_dict(loaded_params, strict=False)

    if args.use_pretrain:
        para_dir = "./res5/gat_pretrain"
        load_epoch = 5
        loaded_params = torch.load(os.path.join(para_dir, f"{category}_{load_epoch}_best_model.pth"))
        filtered_params = {k: v for k, v in loaded_params.items() if 'classifier' not in k}
        model.load_state_dict(filtered_params, strict=False)

    
    epochs = args.epoch
    best_miou_val = -1
    plotter = TrainingMetricsPlotter(save_dir=args.output_dir)
    
    metrics_data = {
        "loss": {
            "train": []
            }, 
        "loss2":{
            "self":[]
        },
        "miou":{
            "train":[], 
            "val":[]
            }
        }
    
    patience_counter = 0
    patience_limit = args.patience_limit
    patience_counter_loss = 0
    min_loss = 0
    if args.mode != "test":
        if args.self_supervised:
                test_indices = random.sample(range(len(dataset_test)), min(len(dataset_test), 8))
                test_subset = Subset(dataset_test, test_indices)
                dataload_self = DataLoader(test_subset, batch_size=bs, shuffle=False)
    
        # opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs if epochs>20 else 20, eta_min=lr / 100)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs//5, eta_min=lr / 100)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)
        
        # opt = AdamW(
        #     [
        #         {"params": [p for name, p in model.named_parameters() if 'classifier' not in name], "lr": 8e-4},  # Exclude classifier
        #         {"params": model.classifier.parameters(), "lr": 8e-3},  # Only classifier here
        #     ],
        #     weight_decay=0.05
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=2e-5)
        
        
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9)
        
        
        params_to_save = {name: param for name, param in model.named_parameters() if param.requires_grad}

        # 加上所有 buffers（例如 BatchNorm 的 running_mean、running_var、num_batches_tracked）
        buffers_to_save = dict(model.named_buffers())
        all_to_save = {**params_to_save, **buffers_to_save}
        
        for epoch in range(epochs):
            io.cprint(f"{category}, Epoch {epoch}/{epochs}")

            if args.self_supervised:
                miou_self, loss_self = run_epoch_self(category, epoch, model, dataload_self, num_label, args, io, writer, optimizer=opt, mode="self", device=device)
                metrics_data["loss2"]["self"].append(loss_self)
            miou_train, loss_train = run_epoch(category, epoch, model, dataload_train, num_label, args, io, writer, optimizer=opt, mode="train", device=device)
            metrics_data["loss"]["train"].append(loss_train)
            metrics_data["miou"]["train"].append(miou_train)
            if min_loss - loss_train < 0.01:
                patience_counter_loss+=1
            else:
                patience_counter_loss = 0
            min_loss = min(min_loss, loss_train)
            
            with torch.no_grad():
                miou_val = run_epoch(category, epoch, model, dataload_val, num_label, args, io, writer, mode="val", device=device)
                metrics_data["miou"]["val"].append(miou_val)
                # writer.add_scalar(f"miou/val", miou_val, epoch)
                if miou_val - best_miou_val < 0.005:
                    patience_counter+=1
                else:
                    patience_counter=0
                    
                if miou_val > best_miou_val:
                    best_miou_val = miou_val
                                        
                    torch.save(all_to_save, best_model_path)
                    io.cprint(f"Saved new best model with validation mIoU: {best_miou_val:.4f}")

            scheduler.step()
        plotter.plot_metrics(metrics_data, title="Training Metrics Overview", save_name=f"{category}_metrics_overview_{seed}.png")
            
            # if args.debug : # or (patience_counter == patience_limit and patience_counter_loss==patience_limit):
            #     break
            
    # Testing
    with torch.no_grad():
        loaded_params = torch.load(best_model_path)
        model.load_state_dict(loaded_params, strict=False)
        test_miou = run_epoch(category, 100000, model, dataload_test, num_label, args, io, writer, mode="test", device=device)
    
    return test_miou, best_miou_val

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
def set_all_seeds_randomly(seed = None):
    # 生成一个随机种子，比如基于当前时间
    if seed is None:
        seed = int(time.time()) % (2**32)
        print(f"Using random seed: {seed}")
    set_all_seeds(seed)
    return seed 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--use_pretrain", type=int, default=0)
    
    
    parser.add_argument("--category", type=str, nargs='+', default=['All']) # ['CoffeeMachine', 'Bottle', 'Cart', 'Refrigerator', 'Laptop', 'Phone']
    parser.add_argument("--shot", type=int, default=8)
    # parser.add_argument("--cuda", type=str, default="cuda:2")
    parser.add_argument("--mode", type=str, default="train")  #!!!!! if change test change output_dir
    parser.add_argument("--output_dir", type=str, default="res/tmp5")
    parser.add_argument("--epoch", type=int, default=20) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience_limit", type=int, default=5)
    
     
    
    parser.add_argument("--use_2d_feat", type=int, default=1) 
    parser.add_argument("--use_3d_feat", type=int, default=0)
    
    parser.add_argument("--img_encoder", type=str, default="dinov2")
    parser.add_argument("--use_cache", type=int, default=0)
    
    parser.add_argument("--sample_pc", type=int, default=0)
    parser.add_argument("--transformer", type=int, default=0)
    parser.add_argument("--conf_bar", type=float, default=0.05)
    parser.add_argument("--use_pseudo_label", type=int, default=0)

    parser.add_argument("--up_method", type=str, default="GA_pooling") # ave GA_pooling
    parser.add_argument("--down_method", type=str, default="MQA_unpooling") # ave MQA_unpooling
    parser.add_argument("--select_edges", type=str, nargs='+', default=["strong","weak"]) # All strong weak
    parser.add_argument("--LH_method", type=str, default="ave") # ave
    
    
    
    parser.add_argument("--use_W_imgfeat", type=int, default=0) # 2
    
    parser.add_argument("--use_propagate", type=int, default=1) # 2
    parser.add_argument("--eliminate_sparseness", type=int, default=1)
    parser.add_argument("--ave_per_mask", type=int, default=0)
    parser.add_argument("--use_gnn", type=int, default=1) 
    
    parser.add_argument("--ave_inter_mask", type=int, default=0)
    
    parser.add_argument("--use_slow_start", type=int, default=-2)
    parser.add_argument("--use_new_classifier", type=int, default=0) 
    parser.add_argument("--use_js2weight", type=int, default=0) 
    parser.add_argument("--use_attn_ave", type=int, default=0)
    parser.add_argument("--use_ave_best", type=int, default=0) 
    parser.add_argument("--back_to_edges", type=int, default=0)
    
    parser.add_argument("--conf_label_edge", type=int, default=0) 
    parser.add_argument("--gt_label_edge", type=int, default=0)
    parser.add_argument("--ps_label_edge", type=int, default=0)
    parser.add_argument("--img_feat_on_mask", type=int, default=0)
    
    parser.add_argument("--All_graph", type=int, default=1)
    parser.add_argument("--use_ball_propagate", type=int, default=0)
    parser.add_argument("--graph4", type=int, default=0)
    
    parser.add_argument("--self_supervised", type=int, default=0)

    parser.add_argument("--use_proxy_contrast_loss", type=int, default=0)
    parser.add_argument("--use_contrast_loss2", type=int, default=0)
    parser.add_argument("--use_ref_loss", type=int, default=0) 
    parser.add_argument("--use_mask_consist_loss", type=int, default=0) 
    parser.add_argument("--use_triplet_loss", type=int, default=0) 
    
    parser.add_argument("--pc_root_path", type=str, default="/raid0/yyhu/dataset/partnetE/PartSLIP/data")
    parser.add_argument("--preprocess_root_path", type=str, default="/data/huyy23/dataset2/PartSLIP/rebuttal_noise/noise_0.01")
    parser.add_argument("--dataloader", type=str, default="PartnetE")
    
    parser.add_argument("--num_view", type=int, default=10)
    
    # CUDA_VISIBLE_DEVICES=2 python train_pc_segmentor_newAllG.py --output_dir ./res/tmp5 --category All --use_propagate 1 --eliminate_sparseness 1 --All_graph 1 --ave_per_mask 1 --up_method ave --down_method ave 
    
    parser.add_argument("--debug", type=int, default=0)
    
    args = parser.parse_args()
    
    print(args.select_edges)
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode=="train":
        file_list=[
            "./train_pc_segmentor_newAllG.py",
            "./scripts/train.sh"
        ]
        copy_files_to_output_dir(file_list, os.path.join(args.output_dir,"code"))
        
        dir_list = [
            "./model",
            "./dataset",
            "./loss"
        ]
        for dir_path in dir_list:
            dest_dir = os.path.join(args.output_dir, "code", os.path.basename(dir_path))
            shutil.copytree(dir_path, dest_dir, dirs_exist_ok=True) 
            
    io = IOStream(os.path.join(args.output_dir, 'run.log'))
    
    io.cprint(str(vars(args)))
    
    writer = None
    
    if not args.pretrain:
        func_train = train

    seed_file_path = os.path.join(args.output_dir, "seed.json")
    seed_record = {}
    
    
    
    if args.category != ["All"] and False:
        io.cprint(str(args.category))
        for category in args.category:
            dataset_test = PartnetEpc("test", category, args)
            dataload_test = DataLoader(dataset_test, batch_size=args.batch_size)
            dataset_train = PartnetEpc("few_shot", category, args)
            dataset_val = PartnetEpc("val", category, args)
            dataload_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
            dataload_val = DataLoader(dataset_val, batch_size=args.batch_size)
            func_train(args, category, io, writer, dataload_train,dataload_val,dataload_test)   

    else:
        if args.category[0] in category_split.keys():
            all_categories = category_split[args.category[0]]
        else:
            if args.category != ["All"]:
                all_categories = args.category
            io.cprint(f"All categories, total {len(all_categories)}")
        print("++++++++++++++++++++++++++++++++++++++")
        print(all_categories)
        print("++++++++++++++++++++++++++++++++++++++")
        for category in all_categories:
            # if category in sp_all_categories:
            #     continue
            dataset_test = dataset_dict[args.dataloader]("test", category, args)
            dataload_test = DataLoader(dataset_test, batch_size=args.batch_size)
            dataset_train = dataset_dict[args.dataloader]("few_shot", category, args)
            dataset_val = dataset_dict[args.dataloader]("val", category, args)
            dataload_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
            dataload_val = DataLoader(dataset_val, batch_size=args.batch_size)
            
            max_best_val_miou = 0
            record_test_miou = 0
            max_best_val_miou_seed = 0
            lines = ""
            for i in range(3):
                seed = set_all_seeds_randomly()
                test_miou, best_val_miou = func_train(args, category, io, writer, dataload_train,dataload_val,dataload_test,seed)
                lines+=f"{test_miou:.4f}, {best_val_miou:.4f}, {seed}\n"
                if best_val_miou>max_best_val_miou:
                    max_best_val_miou = best_val_miou
                    record_test_miou = test_miou
                    max_best_val_miou_seed = seed
            io.cprint(f"{category}:\n {lines}")
            seed_record[category]=(seed, max_best_val_miou.item(), record_test_miou.item())
            with open(seed_file_path, 'w') as f:
                json.dump(seed_record, f)

#  ps -p 2539236 -o args=
# python train_pc_segmentor_newAllG.py --output_dir ./res9/GA_MQA_strong_weak2_3max --category All --debug 0