import os
# CUDA_VISIBLE_DEVICES=0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from tqdm.auto import tqdm
import json
import shutil

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

from src.utils import get_seg_color, copy_files_to_output_dir, IOStream
from src.draw import TrainingMetricsPlotter
# from dataset.PartnetEpc import PartnetEpc
from dataset.PartnetEpcAllG import PartnetEpc
from model.SegmentorNewAllG import SegmentorNew
from scipy.stats import mode as sci_mode

# all_categories = ['Phone','CoffeeMachine','Laptop',  'Bottle',  'Cart', 'Refrigerator',  'Box', 'Bucket', 'Camera', 'Chair', 'Clock', 'Dishwasher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Pliers', 'Printer', 'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window', 'StorageFurniture']
# all_categories = ['Phone', 'CoffeeMachine', 'Laptop', 'Cart', 'Refrigerator', 'Bucket', 'Camera', 'Chair', 'Dishwasher', 'Door', 'Faucet', 'Kettle', 'KitchenPot', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Printer', 'Remote', 'Safe', 'Scissors', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'WashingMachine', 'StorageFurniture']
all_categories = ['Phone', 'CoffeeMachine', 'Laptop', 'Cart', 'Refrigerator', 'Bucket', 'Camera', 'Chair', 'Dishwasher', 'Door', 'Faucet', 'Kettle', 'KitchenPot', 'Lamp', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Printer', 'Remote', 'Safe', 'Scissors', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'WashingMachine', 'StorageFurniture']
# small_part = ["handle", "button", "wheel", "knob", "switch", "bulb", "shaft", "touchpad", "camera", "screw","handlebar","trigger"]
small_part = ['bulb', 'button', 'camera', 'handle', 'knob', 'screw', 'shaft', 'switch', 'touchpad', 'wheel']

def get_legend(img, img_label, label_name):
    height = 20
    legend_height = 100
    unique_labels = np.unique(img_label)
    for i, label in enumerate(unique_labels):
        if label == 0:  # 跳过背景类
            continue
        ind = img_label==(label)
        if ind.sum()==0:
            continue
        color = img[ind][0]
        name = label_name[label]
        # 设置颜色条的起始和结束位置
        start_y = height + (i * (legend_height // len(unique_labels)))  # 图例的纵向位置
        end_y = start_y + (legend_height // len(unique_labels))
        
        # 在图例区域绘制颜色条
        cv2.rectangle(img, (10, start_y), (50, end_y-5), color.tolist(), -1)
        
        # 添加类别标签文字
        cv2.putText(img, f"{name}", (55, start_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def check(args, category, view, img, label, pred, sam_masks_label, num_label, label_name, ccnt=0, miou=0, seed=None):
    img = img.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    sam_masks_label = sam_masks_label.detach().cpu().numpy()
    save_dir = f"output2/{args.output_dir}/{args.mode}_check_{category}_{seed}/check{ccnt}_{str(miou)}"
    print(save_dir)
    # save_dir = f"output/check"  
    os.makedirs(save_dir, exist_ok=True)
    tmp_label = ["BK","None"]
    tmp_label.extend(label_name)
    label_name = tmp_label
    
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f"{save_dir}/output_{view}_image.png", img)
    
    label = label.reshape(-1)
    rgb = get_seg_color(label, num_label)
    rgb[label==0]=0
    rgb = rgb.reshape(800,800,3)
    rgb  = (rgb  * 255).astype(np.uint8) 
    rgb = get_legend(rgb, label.reshape(800,800), label_name)
    cv2.imwrite(f"{save_dir}/output_{view}_label.png", rgb)
    
    pred = pred.reshape(-1)
    rgb = get_seg_color(pred, num_label)
    rgb[label==0]=0
    rgb = rgb.reshape(800,800,3)
    rgb  = (rgb  * 255).astype(np.uint8) 
    rgb = get_legend(rgb, pred.reshape(800,800), label_name)
    cv2.imwrite(f"{save_dir}/output_{view}_pred.png", rgb)
    
    sam_masks_label = sam_masks_label.reshape(-1)
    sam_masks_label = sam_masks_label+1
    sam_masks_label[label==0]=0
    rgb = get_seg_color(sam_masks_label, sam_masks_label.max()+1)
    rgb[label==0]=0
    rgb = rgb.reshape(800,800,3)
    rgb  = (rgb  * 255).astype(np.uint8) 
    cv2.imwrite(f"{save_dir}/output_{view}_mask.png", rgb)
    pass

def calc_miou(pred, target, num_classes):
    
    miou = 0
    cnt = 0
    part_miou = [[] for _ in range(num_classes)]
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
            part_miou[i].append((I/U).cpu().data.numpy())
    if cnt==0:
        return -1
    miou /= cnt
    return miou.cpu().data, part_miou

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
            
def run_epoch(category, epoch, model, dataloader, num_label, args, io, optimizer=None, mode="train", device=None, seed=None):
    is_training = (mode == "train")
    if is_training:
        model.train()
    else:
        model.eval()

    loss_epoch, miou_list, miou_vote_list = [], [], []
    part_miou_list = [[] for _ in range(num_label)]
    # if args.save_img:
    #     dt = len(dataloader)//20
    #     dt = max(dt, 1)
    for idx, batch in enumerate(tqdm(dataloader)):
        # if args.save_img and idx%dt!=0:
        #     continue
        # torch.cuda.empty_cache()
        # if idx!=4:
        #     continue
        pc_id, pc, pc_label, unseen_pc, unseen_pc_label, img, mask_label, pc_idx, coords, pc_fpfh, pc_norm = batch
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

        if mode == "train":
            input_pc_id = pc_id
        elif mode == "val":
            input_pc_id = pc_id + 10
        elif mode == "test":
            input_pc_id = -1
        
        logits, loss, n_pc_label, _ = model(input_pc_id, pc, pc_label, img, mask_label, pc_idx, coords, pc_fpfh, pc_norm, args=args, epoch=epoch, true_pc_id=pc_id)
        predicted = logits.argmax(dim=1)
    
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            
        miou, part_miou = calc_miou(predicted, n_pc_label, num_label)
        for i in range(num_label):
            part_miou_list[i].extend(part_miou[i])
        # predicted[vaoid_votes] = pseudo_label[vaoid_votes]
        # miou_vote = calc_miou(predicted, n_pc_label, num_label)
        # if miou!=-1:
        #     miou_list.append(miou)
        #     miou_vote_list.append(miou_vote)
            
        

        
        if args.save_img:
            partnete_meta = json.load(open("PartNetE_meta.json"))
            view_num = pc_idx.shape[0]
            invalid_ind = pc_idx==-1
            img_label = pc_label[pc_idx]+1
            img_label[invalid_ind]=0
            img_pred = predicted[pc_idx]+1
            img_pred[invalid_ind]=0
    
            for view in range(view_num):
                # print("miou: ", miou)
                check(args, category, view, img_ori[view], img_label[view], img_pred[view], mask_label[view],num_label+1, partnete_meta[category], idx, miou, seed)
                print(idx, miou)
            # if idx == 100 or miou < 0.54:
            #     print(miou)
            #     exit(0)
            # else:
            #     print(miou)
            # print("good")
        
        

    miou_mean = np.mean(miou_list)
    if is_training:
        mean_loss = np.mean(loss_epoch)
        io.cprint(f"{mode.capitalize()} mIoU {category} : {miou_mean:.4f}, Loss: {mean_loss:.4f}")
        return miou_mean, part_miou_list, mean_loss
    else:
        io.cprint(f"{mode.capitalize()} mIoU {category} : {miou_mean:.4f}")
        return miou_mean, part_miou_list

def train(args, category, io):
    bs = args.batch_size
    lr = 0.003
    partnete_meta = json.load(open("PartNetE_meta.json"))
    
    num_label = len(partnete_meta[category]) + 1  
    seed = json.load(open(os.path.join(args.output_dir,"seed.json")))[category]["seed"]
    best_model_path = os.path.join(args.output_dir, f"{category}_best_model_{seed}.pth")
    model = SegmentorNew(num_labels=num_label, args=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Testing
    with torch.no_grad():
        dataset_test = PartnetEpc(args.mode, category, args, show_figure=False)
        dataload_test = DataLoader(dataset_test, batch_size=bs)
        
        
        loaded_params = torch.load(best_model_path)
        model.load_state_dict(loaded_params, strict=False)
        
        miou, part_miou_list = run_epoch(category, -1, model, dataload_test, num_label, args, io, mode="test", device=device, seed=seed)
        return part_miou_list
    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json"))
    # small_category = []
    # spart = []
    # for category in all_categories:
    #     Jud = 0
    #     for part in partnete_meta[category]:
    #         if part in small_part:
    #             print(category, part)
    #             spart.append(part)
    #             Jud = 1
    #     if Jud:
    #         small_category.append(category)
    # spart = np.unique(spart)
    # print(spart)
    
    # print(len(small_category))
    # print("small category: ", small_category)
    # exit(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--use_pretrain", type=int, default=0)
    
    parser.add_argument("--category", type=str, nargs='+', default=["All"] )# ,"Phone", "Bucket", "CoffeeMachine","Display","Safe","Box","Lighter","USB","Clock","Pen"]) # ['Phone', 'Mouse','Switch','Camera','USB','Pen']
    # parser.add_argument("--cuda", type=str, default="cuda:2")
    parser.add_argument("--mode", type=str, default="test")  #!!!!! if change test change output_dir
    parser.add_argument("--shot", type=int, default=8) 
    parser.add_argument("--output_dir", type=str, default="res_fin/GA_MQA_Graph_3max")
    parser.add_argument("--epoch", type=int, default=20) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience_limit", type=int, default=5) 
    parser.add_argument("--transformer", type=int, default=0)
    
    
    parser.add_argument("--use_2d_feat", type=int, default=1) 
    parser.add_argument("--use_3d_feat", type=int, default=0)
    
    parser.add_argument("--img_encoder", type=str, default="dinov2")
    parser.add_argument("--use_cache", type=int, default=0)
    parser.add_argument("--sample_pc", type=int, default=0)
    parser.add_argument("--conf_bar", type=float, default=0.05)
    
    parser.add_argument("--up_method", type=str, default="GA_pooling") # ave GA_pooling
    parser.add_argument("--down_method", type=str, default="MQA_unpooling") # ave MQA_unpooling
    parser.add_argument("--select_edges", type=str, default=["strong","weak"]) # All strong weak
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
    parser.add_argument("--use_pseudo_label", type=int, default=0)
    
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
    
    
    parser.add_argument("--debug", type=int, default=1)
    
    parser.add_argument("--result_name", type=str, default="test.log")
    parser.add_argument("--save_img", type=int, default=0)
    parser.add_argument("--save_psd_label", type=int, default=0)
    parser.add_argument("--num_view", type=int, default=10)
    
    
    args = parser.parse_args()
    
    # if args.ave_per_mask:
    #     args.output_dir = args.output_dir + "_ave"
    # if args.sam_mask_vote:
    #     args.output_dir = args.output_dir + "_vote"
    os.makedirs(args.output_dir, exist_ok=True)
            
    io = IOStream(os.path.join(args.output_dir, args.result_name))
    
    io.cprint(str(vars(args)))
    
    if args.category != ["All"]:
        io.cprint(str(args.category))
        for category in args.category:
            train(args, category, io)
    else:
        io.cprint(f"All categories, total {len(all_categories)}")
        for category in all_categories:
            part_miou = train(args, category, io)
            