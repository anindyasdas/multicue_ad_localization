import numpy as np
import torch
import torch.nn as nn
import random
import argparse
import os
from copy import deepcopy
from scipy import interpolate
from sklearn.cluster import KMeans, MiniBatchKMeans
#from dataloaders.dataloadert1 import build_dataloader #for diffuion
#from dataloaders.dataloadert2 import build_dataloader #for anodiffuison
from dataloaders.dataloadert import build_dataloader #for original perlin based
from modeling.netm1 import SemiADNet
from tqdm import tqdm
from utils import aucPerformance, run_k_means, get_loss_weights, seed_everything
from loss import FocalLoss
import pickle
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


def log_entropy_anomaly_scorer(entropy, epsilon=1e-8):
    """Returns log(1 + entropy) as the anomaly score (stabilized entropy)."""
    return np.log(1 + entropy + epsilon)


def statfree_scale(x, c, D, eps=1e-12, pre=None):
    """
    Monotonic, stat-free map to [0,1]:  s(x) = log1p(min(x,D)/c) / log1p(D/c)
    - c: scale knob (smaller → more spread near 0)
    - D: upper cap (keeps extremes from dominating)
    - pre: optional 'sqrt' to expand tiny values before scaling
    """
    x = np.asarray(x)
    if pre == 'sqrt':
        x = np.sqrt(np.maximum(x, 0.0))
    x = np.minimum(np.maximum(x, eps), D)
    return np.log1p(x / c) / np.log1p(D / c)


def transform_scores1(dev_raw, ent_raw, seg_raw):
    # dev: compress tail
    dev_t = np.log1p(np.maximum(dev_raw, 0.0))

    # ent: you already did log(1+entropy) before passing here
    ent_t = ent_raw

    
    gamma = 0.5  # sqrt to expand
    seg_t = np.power(np.maximum(seg_raw, 0.0), gamma)
    return dev_t, ent_t, seg_t

def transform_scores2(dev_raw, ent_log1pH, seg_raw, eps=1e-12):
    """
    One-shot, deterministic mapping of each signal to [0,1] without any data-dependent fitting.
    dev_raw: deviation (>=0), heavy-tailed
    ent_log1pH: already log(1+H), with H in [0, ln 2] -> ent in [0, ~0.526]
    seg_raw: mean top-k anomaly prob in [0,1] but small (e.g., 0.05–0.22)
    """
    # Deviation: compress tail and scale
    dev_t = np.log1p(np.maximum(dev_raw, 0.0))
    dev_s = np.clip(dev_t / 3.0, 0.0, 1.0)  # 3.0 is a fixed constant; bump to 3.5 if devs get larger

    # Entropy: scale by theoretical max of log(1 + H), H<=ln2
    #ENT_MAX = np.log(2.0)  # ≈ 0.526
    ENT_MAX = np.log(1.0 + np.log(2.0)) #max value is when prob is 0.5, -> ln2
    
    ent_s = np.clip(ent_log1pH / (ENT_MAX + eps), 0.0, 1.0)

    # Segmentation: stretch small values
    seg_s = np.clip(2.0 * np.sqrt(np.maximum(seg_raw, 0.0)), 0.0, 1.0)

    return dev_s, ent_s, seg_s

def transform_scores(dev_raw, ent_input, seg_raw):
    #ENT_MAX = np.log(2.0)  # ≈ 0.526
    ENT_MAX= log_entropy_anomaly_scorer(np.log(2.0)) #max value is when prob is 0.5, -> ln2
    ent_s = statfree_scale(ent_input, c=ENT_MAX/10.0, D=ENT_MAX)

    dev_s = statfree_scale(dev_raw, c=0.05, D=1.0)
    seg_s = statfree_scale(seg_raw, c=0.05, D=1.0, pre='sqrt')
    return dev_s, ent_s, seg_s


def save_obj(my_object, file_name='myfile.pkl'):
    with open(file_name, "wb") as pfile:
        pickle.dump(my_object, pfile)
def compute_soft_prob(y_pred, prob, temperature=5, center=0.5, max_val=0.95):
    # Clone to avoid modifying y_pred
    soft_prob = torch.ones_like(y_pred)  # default to 1 for prob == 1

    # Mask for uncertain samples
    mask = (prob == 0)

    if mask.any():
        # Extract uncertain values
        y_pred_uncertain = y_pred[mask]

        # Min-max normalize only uncertain values
        min_val = y_pred_uncertain.min()
        max_val_ = y_pred_uncertain.max()
        y_pred_norm = (y_pred_uncertain - min_val) / (max_val_ - min_val + 1e-6)

        #print("Normalized (uncertain) y_pred:", y_pred_norm.min(), y_pred_norm.max())

        # Apply non-linear squashing
        y_pred_boosted = torch.sigmoid(temperature * (y_pred_norm - center))
        y_pred_boosted = y_pred_boosted * max_val

        # Fill back only masked positions
        soft_prob[mask] = y_pred_boosted

        #print("y_pred_boosted:", y_pred_boosted)

    #print("soft_prob:", soft_prob)
    #print("prob:", prob)

    return soft_prob




    
class SoftDeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def get_losses(self, y_pred):
        confidence_margin = 10.
        ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        return inlier_loss, outlier_loss
    

    def forward(self, y_pred, prob):
        inlier_loss, outlier_loss = self.get_losses(y_pred)
        total_loss= (1-prob.detach())*inlier_loss+ (prob.detach())*outlier_loss
        #gamma=5   #range 1 to 5 strength
        #total_loss= torch.pow((1-prob.detach()), 1/gamma)*inlier_loss+ torch.pow((prob.detach()), gamma)*outlier_loss
        #total_loss= torch.pow((1-prob.detach()), 0.5)*inlier_loss+ (prob.detach())*outlier_loss
        return total_loss
        
##############################################################################       ############### 
class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.epochs=[5,7,10,15,17,20,22,24]
        self.info_dict_train=dict()
        self.info_dict_test=dict()
        self.weights_candidates = [
            # Relative domination
            #(0.7, 0.2, 0.1), (0.7, 0.1, 0.2),
            

            

            # One dominates, two equal
            #(0.6, 0.2, 0.2),  
            #(0.80, 0.10, 0.10),

            

            #Few Extra
            #(0.60, 0.10, 0.30), (0.55, 0.10, 0.35), (0.58, 0.10, 0.32), (0.65, 0.10, 0.25), (0.60, 0.15, 0.25), (0.50, 0.15, 0.35),
            #(0.75, 0.10, 0.15), (0.80, 0.08, 0.12),
            
            (0.55, 0.10, 0.35),
            

            # Two equal but dominate
            #(0.4, 0.4, 0.2), (0.4, 0.2, 0.4), 
            # All equal
            #(0.33, 0.33, 0.33),
            #extreme Check
            #(1.0, 0, 0)

            ]


        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = build_dataloader(args, **kwargs)

        self.model = SemiADNet(args)
        
        
        
        #Code for instance re-weighting
        self.total_iter= len(self.train_loader) *self.args.epochs
        #burn-in (initial training with uniform weight) 
        self.burnin= int(self.total_iter *0.06)
        self.curr_iter=0
        self.burnin_interp_fn = interpolate.interp1d([self.burnin, self.burnin *3, self.total_iter],
                      [self.args.lambda_hyp * 10, self.args.lambda_hyp, self.args.lambda_hyp])
        

        self.criterion= SoftDeviationLoss()
        self.criterion2= nn.BCELoss(reduce=False)
        #self.criterion2= nn.BCEWithLogitsLoss(reduction='none')
        self.criterion3= FocalLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()
           self.criterion2=self.criterion2.cuda()
           self.criterion3=self.criterion3.cuda()
    

    def train(self, epoch):
        train_loss = 0.0
        train_segment_loss=0.0
        train_abnormal_loss=0.0
        train_ce_loss=0.0
        anomaly_list=[]
        labels_list=[]
        probs=[]
        self.model.train()
        tbar = tqdm(self.train_loader)
        if epoch >=0:
            self.correct_label=False
            
        for i, sample in enumerate(tbar):
            image, anomaly_mask, target, target_true = sample['image'], sample['mask'], sample['label'], sample['true_label']
            
            if self.args.cuda:
                image, anomaly_mask, target, target_true = image.cuda(), anomaly_mask.cuda(), target.cuda(), target_true.cuda()
            
            out_mask, outputs, prob = self.model(image)
            #prob=F.sigmoid(prob_logits)
            
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            
                
            
            segment_loss = self.criterion3(out_mask_sm, anomaly_mask)
            
            #########approximation of lambda_hyp based on iteration###############
            if self.burnin > 0 and self.curr_iter > self.burnin:
                lambda_hyp = self.burnin_interp_fn(self.curr_iter)
                #if np.random.rand() >=0.7:
                #    soft_prob=compute_soft_prob(outputs.detach(), target.unsqueeze(1).float())
                #    losses=self.criterion(outputs, soft_prob.detach().float()).view(-1, 1)
                #else:
                #    losses=self.criterion(outputs, prob.detach().float()).view(-1, 1)
                soft_prob=compute_soft_prob(outputs.detach().float(), target.unsqueeze(1).float())
                
                
                losses=self.criterion(outputs, prob.detach().float()).view(-1, 1) 
                #cluster_labels=run_k_means(outputs).cuda()
                
                combined_labels= (soft_prob.squeeze() + target).cpu()
                #combined_labels= soft_prob.squeeze().cpu()
                nor_index =np.argwhere(combined_labels>= 1.0).flatten()
                combined_labels[nor_index]=1
                combined_labels=combined_labels.cuda()
                
                if np.random.rand() >=0.4:
                    CE_losses= self.criterion2(prob.squeeze(), combined_labels.float())
                else:
                    CE_losses= self.criterion2(prob.squeeze(), target.float())
                #CE_losses= self.criterion2(prob.squeeze(), target.float())
                #CE_losses= self.criterion2(prob_logits.squeeze(), target.float())
                #CE_losses= self.criterion2(prob.squeeze(), combined_labels.float())
                
                
                
               
                
            else:
                lambda_hyp = self.args.lambda_hyp
                losses=self.criterion(outputs, target.unsqueeze(1).float()).view(-1, 1)
                CE_losses= self.criterion2(prob.squeeze(), target.float())
                #CE_losses= self.criterion2(prob_logits.squeeze(), target.float())
                
            
            
            
            batch_size=CE_losses.size()[0]
            
            
            loss_weights1= torch.from_numpy(get_loss_weights(CE_losses.detach().cpu().numpy(), self.args.div_type, self.args.alpha, 
                        self.args.lambda_hyp, self.args.w_type, self.curr_iter,
                     self.burnin))
            loss_weights2= torch.from_numpy(get_loss_weights(losses.detach().cpu().numpy(), self.args.div_type, self.args.alpha, 
                        self.args.lambda_hyp, self.args.w_type, self.curr_iter,
                     self.burnin))
            
            if self.args.cuda:
                loss_weights1=loss_weights1.cuda()
                loss_weights2=loss_weights2.cuda()

            CE_loss=torch.mean(CE_losses*loss_weights1)
            loss=torch.mean(losses*loss_weights2)
            #CE_loss=torch.mean(CE_losses)
            #loss=torch.mean(losses)
            self.optimizer.zero_grad()
            total_loss= loss + CE_loss + segment_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.curr_iter+=1

            anomaly_list.extend(outputs.data.cpu().numpy().tolist())
            labels_list.extend(target_true.cpu().numpy().tolist())
            probs.extend(prob.squeeze().detach().cpu().numpy().tolist())
           
            train_segment_loss += segment_loss.item()
            train_abnormal_loss += loss.item()
            train_loss +=loss.item()
            train_ce_loss += CE_loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.5f Train segment loss: %.5f Train loss ce: %.5f' % (epoch, train_loss / (i + 1), train_segment_loss / (i + 1), train_ce_loss / (i + 1)))
        self.scheduler.step()
        if (epoch+1) in self.epochs:
                self.info_dict_train[epoch]={"anomaly_scores":anomaly_list, "labels":labels_list, "probs":probs}

    def eval(self, data_loader, print_results=True):
        self.model.eval()
        tbar = tqdm(data_loader, desc='\r')
        for weights in self.weights_candidates:
            w_dev, w_ent, w_seg= weights
            test_loss = 0.0
            total_target = np.array([])
            total_pred = list()
            deviation_score=list()
            entropy_score=list()
            segmentation_score=list()
            #total_pred_from_mask=list()
            anomaly_score_prediction = []
            
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    
                    out_mask, outputs, prob = self.model(image.float())
                    
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                    #prob=F.sigmoid(prob_logits)
                    losses=self.criterion(outputs, prob.detach().float()).view(-1, 1)
                    
                    loss=torch.mean(losses)


                ##########################Entropy############################
                prob_clamped = prob.detach().clamp(min=1e-6, max=1 - 1e-6)
                entropy = -prob_clamped * torch.log(prob_clamped) - (1 - prob_clamped) * torch.log(1 - prob_clamped)
                
                # Adjust anomaly scores
                #adjusted_outputs = outputs.detach() * entropy
                
                anomaly_map = out_mask_sm[:, 1, :, :]  # shape: (B, H, W)
                B, H, W = anomaly_map.shape
                anomaly_map_flat = anomaly_map.view(B, -1)
                topk = max(int(H * W * 0.1), 1)
                topk_scores, _ = torch.topk(anomaly_map_flat, topk, dim=1)
                image_scores_seg = torch.mean(topk_scores, dim=1)  # shape: (B,)

                #weights = anomaly_map_flat / (anomaly_map_flat.sum(dim=1, keepdim=True) + 1e-6)
                #image_scores_seg = torch.sum(anomaly_map_flat * weights, dim=1)

                #binary_map = (anomaly_map > 0.5).float()  # or any tuned threshold
                #image_scores_seg = binary_map.view(B, -1).mean(dim=1)  # % of anomalous pixels

                #binary_map = (anomaly_map > 0.5).float()
                #extent = binary_map.view(B, -1).mean(dim=1)  # fraction of anomalous area
                #adjusted_outputs = outputs.view(-1, 1) * entropy.view(-1, 1) * image_scores_seg.view(-1, 1) * extent.view(-1, 1)

                score1=outputs.detach().cpu().numpy()
                score2= entropy.cpu().numpy()
                score2=log_entropy_anomaly_scorer(score2)
                score3=image_scores_seg.cpu().numpy()
                score1, score2, score3=transform_scores(score1, score2, score3)
                #adjusted_outputs = 0.5*score1.reshape(-1, 1)*10 + 0.25*score2.reshape(-1, 1)*10 + 0.25*score3.reshape(-1, 1)*10
                #adjusted_outputs = score1.reshape(-1, 1)*score2.reshape(-1, 1)*score3.reshape(-1, 1)
                eps = 1e-6
                #w_dev, w_ent, w_seg = 0.5, 0.25, 0.5
                

            
                w_sum = (w_dev + w_ent + w_seg)
                logp = w_dev * np.log(score1.reshape(-1,1) + eps) + w_ent * np.log(score2.reshape(-1,1) + eps) + w_seg * np.log(score3.reshape(-1,1) + eps)
                adjusted_outputs = np.exp(logp / w_sum).reshape(-1,1)
                #adjusted_outputs = score1.reshape(-1, 1)*score3.reshape(-1, 1)

                            # ---- Raw scores ----
                

                

                #adjusted_outputs = (np.log(score1.reshape(-1, 1)) +10)* score2.reshape(-1, 1) *10* np.log(1+score3.reshape(-1, 1))*10
                #adjusted_outputs =  score2.reshape(-1, 1)
                #adjusted_outputs = np.log(score1.reshape(-1, 1)) * score2.reshape(-1, 1)
                #adjusted_outputs = score3.reshape(-1, 1)
                #adjusted_outputs=(np.log(score1.reshape(-1, 1))+10)*score2.reshape(-1, 1)*10*np.log(1+seg_score)*10
                ##################################################################

                #out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                #                                                   padding=21 // 2).cpu().detach()
                #out_mask_averaged = out_mask_averaged.view(int(out_mask_averaged.size(0)), -1)
                #topk = max(int(out_mask_averaged.size(1) * 0.1), 1)
                #image_score = torch.mean(out_mask_averaged, dim=1).view(-1,1)
                
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                total_target = np.append(total_target, target.cpu().numpy())
                
                #data = outputs.data.cpu().numpy()
                #data = adjusted_outputs.cpu().numpy()
                data = adjusted_outputs #.cpu().numpy()
                
                total_pred = np.append(total_pred, data)
                deviation_score=np.append(deviation_score, score1)
                entropy_score=np.append(entropy_score, score2)
                segmentation_score=np.append(segmentation_score, score3)
                #total_pred_from_mask=np.append(total_pred_from_mask, image_score.numpy())
            roc, pr=0,0
            if print_results:
                roc, pr = aucPerformance(total_pred, total_target)
            else:
                self.info_dict_test={"anomaly_score":total_pred, "deviation_score":deviation_score, "entropy_score":entropy_score, "segmentation_score":segmentation_score, "labels":total_target}

            if print_results:
                with open(self.args.report_name + ".txt", 'a') as f:
                    f.write("Class: %s, alpha: %.4f, lambda:%.4f, Contamination: %.4f, "
                    " AUC-ROC: %.4f, AUC-PR: %.4f , Weights (dev, ent, seg) : (%.3f, %.3f, %.3f)\n" % (self.args.classname, self.args.alpha, 
                    self.args.lambda_hyp, self.args.cont, roc, pr, w_dev , w_ent , w_seg))
        return

    def save_weights(self, filename):
        #model_wt_name = self.args.dataset +'_' +self.args.classname + '_' + str(self.args.cont) +'_'+ filename
        model_wt_name = filename
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, model_wt_name))


from torch.utils.data import Dataset, DataLoader

class DictTensorDataset(Dataset):
    """Unify train+test tensors into dict samples."""
    def __init__(self, images, masks, labels, true_labels):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.true_labels = true_labels

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'mask': self.masks[idx],
            'label': self.labels[idx],
            'true_label': self.true_labels[idx],
        }


@torch.no_grad()
def make_combined_loader(train_loader, test_loader, batch_size,
                         shuffle=True, num_workers=4, pin_memory=False, drop_last=False):
    imgs, masks, labels, tlabels = [], [], [], []

    # 1) Train loader → keep only true_label == 0
    for sample in train_loader:
        keep = (sample['true_label'] == 0)
        imgs.append(sample['image'][keep].cpu())
        masks.append(sample['mask'][keep].cpu())
        labels.append(sample['label'][keep].cpu())
        tlabels.append(sample['true_label'][keep].cpu())

    # 2) Test loader → keep all, but no mask/true_label → fill defaults
    for sample in test_loader:
        imgs.append(sample['image'].cpu())
        labels.append(sample['label'].cpu())

        B = sample['image'].size(0)
        # fill dummy mask (zeros) and dummy true_label (=label)
        masks.append(torch.zeros(B, 1, sample['image'].size(2), sample['image'].size(3)))
        tlabels.append(sample['label'].cpu())   # treat label as true_label

    # Concatenate
    images = torch.cat(imgs, dim=0)
    masks = torch.cat(masks, dim=0)
    labels_ = torch.cat(labels, dim=0)
    true_labels = torch.cat(tlabels, dim=0)

    dataset = DictTensorDataset(images, masks, labels_, true_labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--dataset', type=str, default='mvtec', help="dataset name")
    parser.add_argument('--anomaly_source_path', type=str, default='./data/dtd/images', help="dataset anomaly source")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=256, help="the image size of input")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    #parser.add_argument('--criterion', type=str, default='deviation-focal', help="the loss function")
    parser.add_argument("--n_anomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--cont", type=float, default=0.05, help="the percentage of contamination")
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha-parameter for alpha divergence')
    parser.add_argument('--div_type',type=str, default='alpha', help='divergence type: alpha divergence')
    parser.add_argument('--lambda_hyp',type=float, default=0.1, help='hyperparameter controlling the radius on the divergence')
    parser.add_argument('--w_type', type=str, default='normalized', help='weight normalization: normalized/unnormalized')
    parser.add_argument('--report_name', type=str, default='result_report_mvtec_bal', help="name of the file where report will be stored")
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.ramdn_seed)
    trainer = Trainer(args)
    
    


    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("class:", args.classname)
    for epoch in range(0, trainer.args.epochs):
        trainer.train(epoch)
    #save_obj(trainer.info_dict_train, file_name=trainer.args.classname +"_" + str(trainer.args.cont) +'_scores_anomaly_train.pkl')
    trainer.eval(trainer.test_loader)
    #comb_loader= make_combined_loader(trainer.train_loader, trainer.test_loader, args.batch_size,
    #                     shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
    #trainer.eval(comb_loader, print_results=False)
    #save_obj(trainer.info_dict_test, file_name=trainer.args.classname +"_" + str(trainer.args.cont) +'_scores_anomaly_test.pkl')
    trainer.save_weights(args.weight_name)
