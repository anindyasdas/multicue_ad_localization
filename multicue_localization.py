import numpy as np
import torch
from modeling.netm1 import SemiADNet
#from datasets import mvtecad
from datasets import mvtecad_perlintest_few, visa_perlintest_few
import cv2
import os
import shutil
import argparse
#from modeling.layers import build_criterion
from utils import aucPerformance
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
from skimage import measure, morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

np.seterr(divide='ignore',invalid='ignore')

def backward_hook(module, grad_in, grad_out):
    #print(f"grad_out: {grad_out[0].shape}")
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    #print(f"fmap:{output.shape}")
    fmap_block.append(output)

def convert_to_grayscale(im_as_arr):
    #print("im_as_arr", im_as_arr.shape) #3, 256, 256
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    #grayscale_im = np.sum(im_as_arr, axis=0)
    #mean = np.mean(grayscale_im)
    #std = np.std(grayscale_im)
    #grayscale_im = (grayscale_im - mean) / std
    
    
    #print("grayscale_im:", grayscale_im.shape) # 256, 256
    im_max = np.percentile(grayscale_im, 99)
    #im_max = np.max(grayscale_im)
    im_min = np.min(grayscale_im)
    if im_max > 0:
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    #print("grayscale_im:", grayscale_im.shape) #1, 256, 256
    return grayscale_im

def show_cam_on_image1(img, mask, label, out_dir, name):

    img1 = img.copy()
    img[:, :, 0] = img1[:, :, 2]
    img[:, :, 1] = img1[:, :, 1]
    img[:, :, 2] = img1[:, :, 0]
    
    #print(mask.shape)
    
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.concatenate((img, cam), axis=1)
    cam = np.concatenate((cam, label), axis=1)

    path_cam_img = os.path.join(out_dir, args.classname + "_cam_" + name + ".jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))


def show_cam_on_image(img, cam, label, out_dir, name):
    """
    Create PaDiM-style visualization with:
    Image | GroundTruth | Predicted heat map | Predicted mask | Segmentation result
    """
    os.makedirs(out_dir, exist_ok=True)

    # normalize and resize
    img = np.uint8(255 * img)
    gt = np.uint8(label * 255)

    # normalize heatmap
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    heat_map = cam * 255
    threshold = 0.5  # or dynamic threshold based on f1 like PaDiM
    mask = cam.copy()
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255

    # segmentation overlay
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

    # plot 5 panels
    fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    vmin, vmax = 0, 255
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for ax_i in ax_img:
        ax_i.axis('off')

    ax_img[0].imshow(img)
    ax_img[0].set_title('Image')

    ax_img[1].imshow(gt, cmap='gray')
    ax_img[1].set_title('GroundTruth')

    ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
    ax_img[2].imshow(img, cmap='gray', interpolation='none')
    ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
    ax_img[2].set_title('Predicted heat map')

    ax_img[3].imshow(mask, cmap='gray')
    ax_img[3].set_title('Predicted mask')

    ax_img[4].imshow(vis_img)
    ax_img[4].set_title('Segmentation result')

    # colorbar
    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Anomaly Score', fontdict={'size': 8})

    fig_img.savefig(os.path.join(out_dir, f"{args.classname}_{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def delete_temp_directory(dirpath):
    #delete the already existing directory
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"Deleting old directory '{dirpath}' .")
        shutil.rmtree(dirpath)
    
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def compute_pro(pred_map, gt_mask, num_thresholds=50):
    """
    Compute PRO (Per-Region Overlap) for one prediction and mask.
    """
    # normalize prediction map to [0,1]
    pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
    thresholds = np.linspace(0, 1, num_thresholds)

    pro_curve = []
    for th in thresholds:
        binary_pred = (pred_map >= th).astype(np.uint8)

        labeled = measure.label(gt_mask, connectivity=2)
        region_props = measure.regionprops(labeled)

        overlaps = []
        for region in region_props:
            coords = region.coords
            region_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            region_mask[tuple(coords.T)] = 1

            inter = np.logical_and(region_mask, binary_pred).sum()
            region_area = region_mask.sum()
            if region_area > 0:
                overlaps.append(inter / region_area)

        if len(overlaps) > 0:
            pro_curve.append(np.mean(overlaps))
        else:
            pro_curve.append(0)

    return np.mean(pro_curve)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--dataset', type=str, default='mvtec', help="dataset name")
    #parser.add_argument('--anomaly_source_path', type=str, default='./data/dtd/images', help="dataset anomaly source")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=256, help="the image size of input")
    parser.add_argument("--n_anomaly", type=int, default=10, help="the number of anomaly data in training set")
    #parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    #parser.add_argument('--criterion', type=str, default='deviation-focal', help="the loss function")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument("--cont", type=float, default=0.05, help="the percentage of contamination")
    #parser.add_argument("--gamma", type=float, default=0.5, help="gamma exponent")
    #parser.add_argument("--beta", type=float, default=0.8, help="beta percentage of focal-deviation loss")
    #parser.add_argument("--cmix_prob", type=float, default=0.4, help="cut_mix percentage for data augmentation")
    #parser.add_argument('--alpha', type=float, default=0.1, help='alpha-parameter for alpha divergence')
    #parser.add_argument('--div_type',type=str, default='alpha', help='divergence type: alpha divergence')
    #parser.add_argument('--lambda_hyp',type=float, default=0.1, help='hyperparameter controlling the radius on the divergence')
    #parser.add_argument('--w_type', type=str, default='normalized', help='weight normalization: normalized/unnormalized')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.ramdn_seed)

    model = SemiADNet(args)
    #model_wt_name = args.dataset +'_'  + args.classname + '_' + str(args.cont)  + '_' + args.weight_name
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, args.weight_name)))
    model = model.cuda()
    model.eval()

    fmap_block = list()
    grad_block = list()
    #print(model)
    model.feature_extractor.net.layer4[1].conv2.register_forward_hook(farward_hook)
    #model.feature_extractor.net.layer4[1].conv2.register_backward_hook(backward_hook)
    
    #print(model)
    if hasattr(model.feature_extractor.net.layer4[1].conv2, "register_full_backward_hook"):
        model.feature_extractor.net.layer4[1].conv2.register_full_backward_hook(backward_hook)
    else:
        model.feature_extractor.net.layer4[1].conv2.register_backward_hook(backward_hook)

    if args.dataset=='mvtec':
        train_set = mvtecad_perlintest_few.MVTecAD(args, train=False)
    elif args.dataset=='visa':
        train_set = visa_perlintest_few.VisaAD(args, train=False)

    outputs = list()
    fmaps = list()
    grads = list()
    seg_label = list()
    outliers_cam = list()
    input = list()
    outlier_scores = list()
    for i in train_set.outlier_idx:
        model.zero_grad()
        sample = train_set.getitem(i)
        inputs = sample['image'].view(1, 3, args.img_size, args.img_size).cuda()
        input.append(np.array(sample['raw_image']))
        inputs.requires_grad = True
        seg, output, prob = model(inputs)
        prob_clamped = prob.clamp(min=1e-6, max=1 - 1e-6)
        entropy = -prob_clamped * torch.log(prob_clamped) - (1 - prob_clamped) * torch.log(1 - prob_clamped)
        
        # Adjust anomaly scores
        #adjusted_outputs = outputs.detach() * entropy
        
        anomaly_map = seg[:, 1, :, :]  # shape: (B, H, W)
        B, H, W = anomaly_map.shape
        anomaly_map_flat = anomaly_map.view(B, -1)
        topk = max(int(H * W * 0.1), 1)
        topk_scores, _ = torch.topk(anomaly_map_flat, topk, dim=1)
        image_scores_seg = torch.mean(topk_scores, dim=1)  # shape: (B,)
        #print(output.shape, entropy.shape, image_scores_seg.shape)
        model.zero_grad(set_to_none=True)
        inputs.grad = None

        output.mean().backward(retain_graph=True)
        entropy.mean().backward(retain_graph=True)
        image_scores_seg.mean().backward()

        #grad = inputs.grad / 3.0   # average the contributions
        #output=output+prob
        #output.backward()
        outlier_scores.append(output.data.cpu().numpy()[0][0])
        #output.backward()
        grad = inputs.grad

        
        #print("input gradient shape:", grad.shape)
        #torch.Size([1, 3, 256, 256])
        grad_temp = convert_to_grayscale(grad.cpu().numpy().squeeze(0))
        #print("gradient temp shape:", grad_temp.shape)
        #(1, 256, 256)
        grad_temp = grad_temp.squeeze(0)
        #grad_temp = gaussian_filter(grad_temp, sigma=8)
        #print("gradient temp shape:2", grad_temp.shape)
        #(256, 256)
        
        ############################### comment
        #outliers_cam.append(grad_temp) #
        ######################################

        outputs.append(output.item())
        fmaps.append(fmap_block)
        total_map=grad_temp
        ###############changes##################
        for grad_map in grad_block:
            tp= torch.mean(torch.abs(grad_map), axis=1)
            #print("tp:", tp.shape)
            score_map = F.interpolate(tp.unsqueeze(0), size=256, mode='bilinear', align_corners=False).squeeze(0)
            #print("1", score_map.shape)
            score_map=convert_to_grayscale(score_map.cpu().numpy())
            score_map = score_map.squeeze(0)
            total_map+=score_map
        #print(score_map.shape)
        total_map= total_map/(len(grad_block)+1)
        #print(np.min(score_map), np.max(score_map))
        total_map = gaussian_filter(total_map, sigma=8)
        outliers_cam.append(total_map)
        
        #############
        grad_block.reverse()
        grads.append(grad_block)
        seg_label.append(np.array(sample['seg_label']))
        fmap_block = list()
        grad_block = list()

    visualization_dir= os.path.join(args.experiment_dir, 'vis' + str(args.cont), args.classname)
    delete_temp_directory(visualization_dir)
    ensure_directory_exists(visualization_dir)

    for i, (cam, raw, label) in enumerate(zip(outliers_cam, input, seg_label)):
        raw = np.float32(cv2.resize(np.array(raw), (args.img_size, args.img_size))) / 255
        label = cv2.resize(label, (args.img_size, args.img_size)) / 255
        show_cam_on_image(raw, cam, label, visualization_dir, str(i))

    aucs = list()
    aucprs=list()
    print("Detected anomaly: " + str(len(outliers_cam)))
    if len(outliers_cam) == 0:
        print('Cannot find anomaly image')
        exit()
    for cam, label in zip(outliers_cam, seg_label):
        label = cv2.resize(label, (args.img_size,args.img_size))
        label = label > 0
        cam_line = cam.reshape(-1)
        #print(label.shape, label[:,:,0].shape)
        label_line = label[:,:,0].reshape(-1)
        auc, aucpr = aucPerformance(cam_line, label_line, prt=False)
        aucs.append(auc)
        aucprs.append(aucpr)
        
        
    aucs = np.array(aucs)
    aucprs = np.array(aucprs)
    print("classname:", args.classname)
    print("Pixel-level AUC-ROC: %.4f" % (np.mean(aucs)))
    print("Pixel-level AUC-PR:  %.4f" % (np.mean(aucprs)))
    print('Visualization results are saved in: ' + os.path.join(args.experiment_dir, 'vis'))