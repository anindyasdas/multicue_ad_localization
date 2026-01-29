import numpy as np
from PIL import Image
import os, sys,shutil
from torchvision import transforms
from copy import deepcopy
import cv2
import glob
import torch
import random
import imgaug.augmenters as iaa
import pandas as pd
import re

from datasets.base_dataset import BaseADDataset
from modeling.transformations.perlin import rand_perlin_2d_np
#from diffusers import StableDiffusionImg2ImgPipeline


def dummy(images, **kwargs):
    return images, [False] * len(images)

class VisaAD(BaseADDataset):

    def __init__(self, args, train = True):
        #super(VisaAD).__init__()
        super().__init__()
        self.args = args
        self.train = train
        self.classname = self.args.classname
        if self.train:
            self.anomaly_source_paths = sorted(glob.glob(self.args.anomaly_source_path+"/*/*.jpg"))
            print(self.args.anomaly_source_path, len(self.anomaly_source_paths))
        

        self.root = os.path.join(self.args.dataset_root, self.classname)
        #self.anomaly_train_dirname= 'anomaly_train_gen_samples'
        self.anomaly_train_dirname= os.path.join(self.classname, 'anomaly_train_gen_samples')
        ####loading data set train test split file for 1 class classification ####
        self.dset_split_file= os.path.join(self.args.dataset_root, 'split_csv', '1cls.csv')
        self.df= pd.read_csv(self.dset_split_file)
        #########################################################
        
        self.transform = [self.transform_train_normal, self.transform_train_outlier] if self.train else [self.transform_test, self.transform_test]

        ######################synthetic contamination diffusion model#########################
        #self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        #    "runwayml/stable-diffusion-v1-5",
        #    torch_dtype=torch.float16
        #)
        # disable safety checker
        #self.pipe.safety_checker = dummy  
        #if self.args.cuda:
        #    self.pipe = self.pipe.to("cuda")#


        ###############################################################################

        
        normal_data = list()
        if self.train:
            split = 'train'
        else:
            split = 'test'
        normal_data=self.df.loc[(self.df['object']==self.classname) & (self.df['split']==split) &(self.df['label']=='normal')]['image'].tolist()
                
        #####compute contamination_size from training data_size, contamination used in training ,not to be used for evaluation during testing########
        normal_train_data=normal=self.df.loc[(self.df['object']==self.classname) & (self.df['split']=='train') &(self.df['label']=='normal')]['image'].tolist()
        self.contamination_size= int((self.args.cont/(1-self.args.cont))*len(normal_train_data))
        print("cont size", self.contamination_size, self.args.cont)
        #fills outlier
        self.get_outlier()
        ###########################################################################################
        
        if split=='train':
            outlier_contamination=self.sample_outliers()
            print(split)
            few_shot_samples = self.outlier_data[:self.args.n_anomaly]
            #print("few_shot_samples", few_shot_samples)
            #print("normal_data:", normal_data)
            print("few-shot_samples:", len(few_shot_samples))
            #print("outlier_contamination", outlier_contamination)
            if len(few_shot_samples)>0:
                few_shot_samples = random.choices(few_shot_samples, k=int(len(normal_data)*1.0))
            print("few-shot_samples oversampled:", len(few_shot_samples))
            print("normal data:", len(normal_data))
            print("outlier_contamination:", len(outlier_contamination))
            #anomaly_data=self.get_outlier()
            #normal_data=normal_data+anomaly_data
            normal_label_true = np.zeros(len(normal_data)).tolist()
            #print("norm:", len(normal_label_true), np.unique(normal_label_true))
            cont_label_true = np.ones(len(outlier_contamination)).tolist()
            #print("cont:", len(cont_label_true), np.unique(cont_label_true))
            normal_data_total=normal_data + outlier_contamination
            normal_label_true=normal_label_true+cont_label_true
            #print("norm2:", len(normal_label_true), np.unique(normal_label_true))
            print("contaminated normal data:", len(normal_data_total))
            pseudo_outlier_data=deepcopy(normal_data_total)
            outlier_data= pseudo_outlier_data + deepcopy(few_shot_samples)
            #outlier_data= deepcopy(normal_data) #Copying normal data to be used for outliers by augmentation
            #2 is the label for pseudo anomaly, 1 is the label for fewshot true anomaly
            outlier_label_true=[2]*len(pseudo_outlier_data) + [1]*len(few_shot_samples)
            #print("out:", len(outlier_label_true), np.unique(outlier_label_true))
            print("outlier_data, available for training including pseudo and few-shot:",  len(outlier_data))
            #print("pseudo outlier_data:",  len(outlier_data))
        else:
            print(split)
            normal_data_total=deepcopy(normal_data)
            print("normal data:",  len(normal_data_total))
            #outlier_data = self.get_outlier()
            outlier_data = self.sample_outliers()[self.args.n_anomaly:]
            print("outlier_data:", len(outlier_data))
            normal_label_true = np.zeros(len(normal_data_total)).tolist()
            outlier_label_true=[1]*len(outlier_data)
            #print("len test outlier", len(outlier_data))
        #outlier_data.sort()

        normal_label = np.zeros(len(normal_data_total)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()

        self.images = normal_data_total + outlier_data
        self.labels = np.array(normal_label + outlier_label)
        self.labels_true = np.array(normal_label_true + outlier_label_true)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()
        
        
        ######augmentation####
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        #sequential augmentation
        self.augmentation = iaa.Sequential([iaa.Flipud(p=0.5), # flip vertically with a probability of 0.5
                      iaa.Fliplr(p=0.5),  # flip horizontally with a probability of 0.5
                      iaa.Affine(rotate=(15, 75)),
                      iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=20))])

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.brightness_contrast_aug=iaa.Sequential([iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30))])
        self.saturation_hue_aug=iaa.AddToHueAndSaturation((-20,20),per_channel=True)
        
        self.transform_img2tensor = transforms.Compose([
            transforms.ToTensor()])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    

    def get_outlier(self):
        outlier_data= self.df.loc[(self.df['object']==self.classname) & (self.df['split']=='test') &(self.df['label']=='anomaly')]['image'].tolist()
        #storing the mask path name in a dictionary, with key as file_id (number)
        #for visa dataset file_id number is same for image and correspoding mask
        img_labels= self.df.loc[(self.df['object']==self.classname) & (self.df['split']=='test') &(self.df['label']=='anomaly')]['mask'].tolist()
        self.img_labels_dict={}
        for path_name in img_labels:
            dir_name, file_name=os.path.split(path_name)
            file_idx=file_name.split('.')[0]
            self.img_labels_dict[file_idx]=path_name
        #if self.contamination_size > len(outlier_data)/2:
        
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        self.outlier_data=outlier_data
        return
    def sample_outliers(self):
        if self.train:
            print("available total anomalies:", len(self.outlier_data))
            outlier_samples= self.synthetic_contamination(self.outlier_data, self.contamination_size)
            return outlier_samples
        else: 
            return self.outlier_data
            
    def synthetic_contamination(self, anorm, num_contamination):
        print("Synthetic anomaly contamination .....")
        anorm_img = [self.load_image(os.path.join(self.args.dataset_root, img_path)) for img_path in anorm]
        #concat accross new axis
        anorm_img= np.stack(anorm_img, axis=0)
        #print(anorm_img.shape)
        try:
            idx_contamination = np.random.choice(np.arange(anorm_img.shape[0]),num_contamination,replace=False)
        except:
            idx_contamination = np.random.choice(np.arange(anorm_img.shape[0]),num_contamination,replace=True)
        train_contamination = anorm_img[idx_contamination]
        #train_contamination = train_contamination + np.random.randn(*train_contamination.shape)*np.std(anorm_img,0,keepdims=True)
        #train_contamination = train_contamination + np.random.randn(*train_contamination.shape)
        train_contamination = train_contamination + np.random.randn(*train_contamination.shape)*2.0
        outlier_path=[]
        dirpath= os.path.join(self.args.dataset_root, self.anomaly_train_dirname)
        self.delete_temp_directory(dirpath)
        self.ensure_directory_exists(dirpath)
        for idx in range(0, train_contamination.shape[0]):
            img_data= train_contamination[idx]
            img_data=self.get_normalized_img(img_data)
            #img_data=self.random_rotation(img_data)
            #print(img_data.shape)
            save_path=os.path.join(self.anomaly_train_dirname, str(idx) + '.jpg')
            outlier_path.append(save_path)
            self.save_image(img_data, os.path.join(self.args.dataset_root, save_path))
        return outlier_path


    """
    def synthetic_contamination1(self, anorm, num_contamination):
        print("Synthetic anomaly contamination .....")
        anorm_img = [self.load_image(os.path.join(self.args.dataset_root, img_path)) for img_path in anorm]
        prompts=[]
        class_name = re.sub(r'[^A-Za-z]', '', " ".join(self.classname.split("_")))
        for img_path in anorm:
            parts=img_path.split(os.sep)
            #defect=" ".join(parts[-2].split("_")) + " defect"
            defect="defect"
            prompts.append(class_name )#+ " but with " + defect)
        
        try:
            idx_contamination = np.random.choice(np.arange(len(anorm_img)),num_contamination,replace=False)
        except:
            idx_contamination = np.random.choice(np.arange(len(anorm_img)),num_contamination,replace=True)
        #train_contamination = anorm_img[idx_contamination]
        chosen_imgs = [anorm_img[idx] for idx in idx_contamination]
        chosen_prompts = [prompts[idx] for idx in idx_contamination]

        
        outlier_path=[]
        dirpath= os.path.join(self.args.dataset_root, self.anomaly_train_dirname)
        self.delete_temp_directory(dirpath)
        self.ensure_directory_exists(dirpath)
        for idx, img_data in enumerate(chosen_imgs):
            #img_pil = Image.fromarray(img_data)
            img_pil = Image.fromarray(img_data).resize((512, 512))
            prompt_=chosen_prompts[idx].lower()
            #print(prompt_)
            result_pil = self.pipe(
                prompt=prompt_,
                image=img_pil,
                strength=0.05, #controls noise
                guidance_scale=10
            ).images[0]
            # convert back PIL → numpy for save_image
            img_data = np.array(result_pil)  # RGB, uint8 [0..255]
            #img_data = np.array(result_pil.resize((self.args.img_size, self.args.img_size)))
            save_path=os.path.join(self.anomaly_train_dirname, str(idx) + '.jpg')
            outlier_path.append(save_path)
            self.save_image(img_data, os.path.join(self.args.dataset_root, save_path))
        return outlier_path
        """
        
        
        
    def delete_temp_directory(self, dirpath):
        #delete the already existing directory
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            print(f"Deleting old directory '{dirpath}' .")
            shutil.rmtree(dirpath)
    
    def ensure_directory_exists(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
        

    def load_image(self, path):
        image = cv2.imread(path)
        # Convert BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, dsize=(self.args.img_size,self.args.img_size))
        #returns numpy array
        return image

    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)
        mask = cv2.resize(mask, (self.args.img_size, self.args.img_size),
                        interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)           # binary 0/1
        mask = np.expand_dims(mask, axis=2)            # (H, W, 1)
        return mask
        
    def random_rotation(self, image):
        p = random.random()
        #random flip while saving
        if p<=0.33: #horizontal
            image = cv2.flip(image, 0) 
        elif p>0.33 and p<=0.66: #vertical
            image = cv2.flip(image, 1)
        return image
        
    def get_normalized_img(self, img_array):
        min_val= np.min(img_array)
        max_val= np.max(img_array)
        img_array=((img_array-min_val)*255.0)/(max_val-min_val)
        return img_array.astype('uint8')
        
    def normalize_to_binary_image(self, img_array):
        """
        Normalize image to [0,255]
        image: 
        returns: PIL image object to [0,255]
        """
        
        min_val= np.min(img_array)
        max_val= np.max(img_array)
        img_array=((img_array-min_val)*255.0)/(max_val-min_val)
        #converting to binary image with threshould 127
        img_array=np.where(img_array < 127, 0, 255)
        image=Image.fromarray(img_array.astype('uint8'))
        return image
        
        
    def save_image(self, image, path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        
    
    def transform_train_normal(self, image, anomaly_source_img=None, mask= None):
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        anomaly_mask=np.zeros((image.shape[0], image.shape[1],1))
        #image = np.transpose(image, (2, 0, 1))
        
        return self.transform_img2tensor(image), self.transform_img2tensor(anomaly_mask)
    
    
        
    def transform_train_outlier(self, image, anomaly_source_img, mask=None):
        #segmask is available for few-shot samples
        #for pseudo anomalies n seg_mask is provided, pseudo anomalies are generated

        if mask is None:
            #pseudo-anomaly
            do_aug_orig = torch.rand(1).numpy()[0] > 0.7
            #Augmentation using perlin noise
            if do_aug_orig:
                image = self.rot(image=image)
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_img)
            #print("train mask is None:", anomaly_mask.shape, np.max(anomaly_mask), np.min(anomaly_mask))
        else:
            #few-shot samples
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            mask=np.asarray(mask)
            augmented_image, anomaly_mask=image, mask
            #print("train mask is not None:", anomaly_mask.shape, np.max(anomaly_mask), np.min(anomaly_mask))
        
       
        #################################################
        return self.transform_img2tensor(augmented_image), self.transform_img2tensor(anomaly_mask)
        
        
    
    def transform_test(self, image, anomaly_source_img=None, mask=None):

        image = image / 255.0
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        if mask is None:
            anomaly_mask=np.zeros((image.shape[0], image.shape[1],1))
        else:
            anomaly_mask=np.asarray(mask)
        
        return self.transform_img2tensor(image), self.transform_img2tensor(anomaly_mask)
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def augment_image(self, image, anomaly_source_img):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        
        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.args.img_size, self.args.img_size), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        
        #if no_anomaly > 0.5:
        if no_anomaly < 0.0:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
        
   
    def __len__(self):
        return len(self.images)

    def getitem(self, index):
        transform_norm, transform_outlier = self.transform
        #transform_norm_compare, transform_outlier_compare = self.transform_compare
        image = self.load_image(os.path.join(self.args.dataset_root, self.images[index]))
        
        if self.labels[index]==0:
            #print("transforming normal", transform_norm)
            #add image compare , for comparision of two variation of same images and maximize the difference
            tr_image, tr_mask=transform_norm(image)
            #print("1", image, image.shape)
            image=Image.fromarray(image.astype('uint8'))
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index], 'raw_image': image, 'seg_label': None}
            #sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index]}
        else:
            #print("transforming outlier", transform_outlier)
            
            if self.train:
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_source_img=self.load_image(self.anomaly_source_paths[anomaly_source_idx])
                image_label= None
            else:
                anomaly_source_img=None
                dir_name, file_name=os.path.split(self.images[index])
                file_idx=file_name.split('.')[0]
                image_label = self.load_image(os.path.join(self.args.dataset_root, self.img_labels_dict[file_idx]))
                #visa dataset mask is normalized, so normalizing to 0, 255
                image_label=self.normalize_to_binary_image(image_label)
            tr_image, tr_mask=transform_outlier(image, anomaly_source_img)
            image=Image.fromarray(image.astype('uint8'))
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index], 'raw_image': image, 'seg_label':image_label}
            #sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index]}
        return sample
    
    def __getitem__(self, index):
        transform_norm, transform_outlier = self.transform
        #transform_norm_compare, transform_outlier_compare = self.transform_compare
        image = self.load_image(os.path.join(self.args.dataset_root, self.images[index]))
        dir_name, file_name=os.path.split(self.images[index])
        file_idx=file_name.split('.')[0]
        seg_mask=None
        
        if self.labels[index]==0:
            #print("transforming normal", transform_norm)
            #add image compare , for comparision of two variation of same images and maximize the difference
            tr_image, tr_mask=transform_norm(image)
            #sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index]}
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index], 'true_label':self.labels_true[index]}
        else:
            #print("transforming outlier", transform_outlier)
            
            if self.train:
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_source_img=self.load_image(self.anomaly_source_paths[anomaly_source_idx])
                
            else:
                anomaly_source_img=None
            if self.labels[index]==1 and self.labels_true[index]==1: #If these are actual anomaly samples (few_shot), not pseudo
                #normal samples both labes 0
                #contamination label 0, true label1
                #pseudo label1, truelabel2
                #fewshot label1 true label
                #print(self.train, "1", self.images[index], self.labels[index] ,self.labels_true[index])
                
                seg_mask = self.load_mask(os.path.join(self.args.dataset_root, self.img_labels_dict[file_idx]))
                #seg_mask=self.normalize_to_binary_image(seg_mask)
                #print(self.train, "2", seg_mask)
            tr_image, tr_mask=transform_outlier(image, anomaly_source_img, mask=seg_mask)
            sample = {'image': tr_image, 'mask':tr_mask, 'label': self.labels[index], 'true_label':self.labels_true[index]}
        return sample

    