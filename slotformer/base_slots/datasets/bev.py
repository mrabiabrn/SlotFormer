import os 
import glob
import json

import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from typing import List, Optional, Sequence, Union, Callable
import matplotlib.pyplot as plt
from .bev_utils import to_rgb


class BEVDataset(Dataset):
    def __init__(self, 
                data_path: str,
                split: str, 
                image_folder: str,
                command_folder: str,   
                video_len: int,         
                skip_frame: int = 1,
                #intra_skip_frame: int = 1,
                n_sample_frames: int = 1,
                #num_out_frame: int = 1,
                transform: Optional[Callable] = None,
                zero_out_red_channel: bool = False,
                stride: int = 1,
            ):

        super(BEVDataset, self).__init__()


        self.data_path = data_path
        self.split = split
        self.image_folder = image_folder
        self.command_folder = command_folder
        self.transform = transform
        self.zero_out_red_channel = zero_out_red_channel
        #self.inter_skip_frame = inter_skip_frame
        self.skip_frame = skip_frame
        self.stride = stride
        self.video_len = video_len

        # num of input and output frames for multistep bev
        self.n_sample_frames = n_sample_frames
        #self.num_out_frame = num_out_frame 
        self.npz_file_name = 'bev'

        self.main_folders = sorted(os.listdir(self.data_path))

        if split=="train":
            self.main_folders = self.main_folders[:-3]  #-10] # TODO: is it for rgb [2:-2]
        elif split=="val":
            self.main_folders = self.main_folders[-3:] #[-9:]  #[self.main_folders[-2]]
        elif split=="test":
            self.main_folders = self.main_folders #[self.main_folders[0]] 
            self.npz_file_name = 'arr_0'

        sub_folder_depth = ''

        self.sub_folders = []
        for folder in self.main_folders:
            sub_folders_path = self.data_path + folder + sub_folder_depth
            self.sub_folders += glob.glob(sub_folders_path)

        self.images = []
        self.commands = []
        self.files = []
        for folder in self.sub_folders:
            img_paths = folder + '/' + self.image_folder + '/*'
            cmd_paths = folder + '/' + self.command_folder + '/*'

            img_paths = sorted(glob.glob(img_paths))
            cmd_paths = sorted(glob.glob(cmd_paths))

            max_frame = len(img_paths) - 1

            # TODO: you will store all paths as strings 
            for t, img in enumerate(img_paths):
                if t%self.stride!=0: #self.split == 'train' and 
                    continue

                if t + self.skip_frame*(self.n_sample_frames-1) <= max_frame:

                    imgs = [img_paths[t + i*self.skip_frame] for i in range(self.n_sample_frames)]
                    #cmds = [] #[cmd_paths[t + i*self.intra_skip_frame] for i in range(self.num_in_frame)]
                    #imgs_out = [img_paths[t + self.intra_skip_frame*(self.num_in_frame-1) + self.inter_skip_frame + i*self.intra_skip_frame] for i in range(self.num_out_frame)]

                    self.files += [(imgs)]
                    #self.files += [(imgs_in,cmds,imgs_out)]


        import random
        # random.seed(13)
        # indices = random.sample(range(0, len(self.files)), 52)
        # print('INDICES ',indices)

        # self.files = [self.files[i] for i in indices]
        # self.files = self.files[:100] 
        #print('files ', self.files[:3])

        print(f"Detected {len(self.files)} images in split {self.split}")
        print(f"Detected {len(self.files)} commands in split {self.split}")

    def __len__(self):

        return len(self.files)


    def get_video(self, video_idx):

        #folder = 'outputs/' #self.files[video_idx]
        file_names = self.files[video_idx]
        print('VIDEO IDX ',video_idx)
        print('FILE NAMES ', file_names)
        num_frames = (self.n_sample_frames + 1) // self.skip_frame - 1 #frame_offset
        #filename = os.path.join('outputs/', str(video_idx), 'test_{}.png')
        #print('FILENAME ', file_names
        
        frames = []
        for n in range(num_frames):
            bev = np.load(file_names[n]).get(self.npz_file_name)

            if bev.shape[2] == 9:
                bev = np.transpose(bev,(2,0,1)) # make channel dim first
            
            bev = np.delete(bev, 2, axis=0)
            bev = np.transpose(bev,(1,2,0))
            bev = self.transform(bev)
            #print('BEVVV ', bev.shape)
            bev[bev>0]=1
            #print('BEVVV ', bev.shape)
            bev = bev[2,:,:].unsqueeze(0)
            #bev = convert_to_one_hot(bev.unsqueeze(0))
            #print('bev ', bev.shape)
            frames.append(bev) #[0]) #to_rgb(bev[0]))
       
        # frames = [
        #     Image.open(filename.format(1 +
        #                                n * self.skip_frame)).convert('RGB') #frame_offset
        #     for n in range(num_frames)
        # ]
        #frames = [self.transform(img) for img in frames]

        return {
            'video': torch.stack(frames, dim=0),
            'data_idx': video_idx,
        }
            
    def __getitem__(self, idx):
        
        imgs = self.files[idx]
        #print('IMGS ', imgs)
        bevs = torch.zeros((self.n_sample_frames,8,192,192)) #192,192))
        #bevs_out = torch.zeros((self.num_out_frame,8,192,192)) #,192,192))

        try:
  
            for i,img_in in enumerate(imgs):

                bev = np.load(img_in).get(self.npz_file_name)

                if bev.shape[2] == 9:
                    bev = np.transpose(bev,(2,0,1)) # make channel dim first
                

                bev = np.delete(bev, 2, axis=0)
                bev = np.transpose(bev,(1,2,0))
                
                if self.transform is not None:
                    bev = self.transform(bev)
                
                bev[bev>0]=1

                bevs[i] = bev 

            # for i,img_out in enumerate(imgs_out):
                
            #     bev = np.load(img_out).get(self.npz_file_name)

            #     if bev.shape[2] == 9:
            #         bev = np.transpose(bev,(2,0,1)) # make channel dim first

            #     bev = np.delete(bev, 2, axis=0)
            #     bev = np.transpose(bev,(1,2,0))

            #     if self.transform is not None:
            #         bev = self.transform(bev)
                
            #     bev[bev>0]=1

            #     bevs_out[i] = bev #torch.from_numpy(bev).float()

        except:
            raise

        bevs = bevs[:,2,:,:].unsqueeze(1) #convert_to_one_hot(bevs)   # TAKE the vehicle channel

        #print('BEV ', bevs.shape)
        #bevs_out = convert_to_one_hot(bevs_out)
        data_dict = {}
        data_dict['data_idx'] = idx
        data_dict['img'] = bevs
        return data_dict #, bevs_out, imgs_in, imgs_out


def convert_to_one_hot(
                        bev, 
                        keep_none_channel=False, 
                        num_channels=8, 
                        dtype=torch.float32
                    ):
    '''
        bev shape --> (T, C, H, W)
    '''
    #print('BEV SHAOE ',bev.shape)
    # flatten the output image to a mask of 1s if the pixel has any of the channels set to 1, 0 otherwise
    none_channel = 1 - bev.max(dim=1, keepdim=True)[0].float()

    # Add an extra channel to the output image
    out_img = torch.cat([bev, none_channel], dim=1)
    # If multiple channels have 1 in the same pixel, we want the index of the highest index channel
    out_img = num_channels - torch.argmax(torch.flip(out_img, [1]), axis=1)

    # convert the output image to one hot encoding
    out_img = torch.nn.functional.one_hot(out_img, num_classes=num_channels + 1)
    out_img = (
        out_img
        .permute(0,len(out_img.shape) - 1, *[i for i in range(1,len(out_img.shape) - 1)]) # permute(0,3,1,2)
        .to(dtype=dtype)
    )

    # If keep_none_channel is False, remove the none channel
    if not keep_none_channel:
        out_img = out_img[:,:-1]

    return out_img


def build_bev_dataset(params, val_only=False):
    """Build BEV video dataset."""
    train_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # [3, H, W]
                                    transforms.CenterCrop(params.
                                    resolution),
                                    ])
    args = dict(
                data_path=params.data_root,
                split='val', 
                transform=train_transforms,
                image_folder=params.image_folder,
                command_folder=params.command_folder,
                skip_frame=params.skip_frame,
                video_len=params.video_len,
                #intra_skip_frame=params.intra_skip_frame,
                n_sample_frames=params.n_sample_frames,
                #num_out_frame=params.num_out_frame,
                stride = params.stride,    
            )
    
    val_dataset = BEVDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = BEVDataset(**args)
    return train_dataset, val_dataset


# class BevLDMDataset(LightningDataModule):
#     """
#     PyTorch Lightning data module 

#     Args:
#         data_dir: root directory of your dataset.
#         train_batch_size: the batch size to use during training.
#         val_batch_size: the batch size to use during validation.
#         patch_size: the size of the crop to take from the original images.
#         num_workers: the number of parallel workers to create to load data
#             items (see PyTorch's Dataloader documentation for more details).
#         pin_memory: whether prepared items should be loaded into pinned memory
#             or not. This can improve performance on GPUs.
#     """

#     def __init__(
#         self,
#         data_path: str,
#         train_batch_size: int = 8,
#         val_batch_size: int = 8,
#         crop_size: Union[int, Sequence[int]] = (192, 192),
#         patch_size: Union[int, Sequence[int]] = (192, 192),
#         num_workers: int = 3,
#         pin_memory: bool = False,
#         image_folder: str = 'bev',
#         command_folder: str = 'measurements',
#         zero_out_red_channel: bool = False,
#         data_type: str = 'rgb',
#         inter_skip_frame: int = 1,
#         intra_skip_frame: int = 1,
#         train_num_in_frame: int = 1,
#         train_num_out_frame: int = 1,
#         val_num_in_frame: int = 1,
#         val_num_out_frame: int = 1,
#         sliding_window_stride: int = 1,
#         test: dict = {},
#         **kwargs,
#     ):
#         super().__init__()

#         print('VAE Dataset data path ', data_path)

#         self.data_path = data_path
#         self.train_batch_size = train_batch_size
#         self.val_batch_size = val_batch_size
#         self.crop_size = crop_size
#         #self.random_crop_chance = random_crop_chance
#         self.patch_size = patch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.image_folder = image_folder
#         self.zero_out_red_channel = zero_out_red_channel
#         self.data_type = data_type
#         self.inter_skip_frame = inter_skip_frame
#         self.intra_skip_frame = intra_skip_frame
#         self.train_num_in_frame=train_num_in_frame
#         self.train_num_out_frame=train_num_out_frame
#         self.val_num_in_frame=val_num_in_frame
#         self.val_num_out_frame=val_num_out_frame
#         self.command_folder = command_folder
#         self.stride = sliding_window_stride
#         self.test = test

#     def setup(self, **kwargs) -> None:

# #       =========================  BEV Dataset  =========================

#         train_transforms = transforms.Compose([   transforms.ToTensor(),
#                                                 #transforms.Resize(192, interpolation=transforms.InterpolationMode.NEAREST),
#                                                 transforms.CenterCrop(192),                                                
#                                                 ])
        
#         val_transforms = transforms.Compose([   transforms.ToTensor(),
#                                                 #transforms.Resize(192, interpolation=transforms.InterpolationMode.NEAREST),
#                                                 transforms.CenterCrop(192),                                                
#                                                 ]) 
        
#         if self.data_type == 'binary':
#             self.train_dataset = BEVBinaryDataset(
#                 data_path=self.data_path,
#                 split='train', 
#                 transform=train_transforms,
#                 image_folder=self.image_folder,
#                 command_folder=self.command_folder,
#                 inter_skip_frame=self.inter_skip_frame,
#                 num_in_frame=self.num_in_frame,
#                 num_out_frame=self.num_out_frame,
#             )
        
#             self.val_dataset = BEVBinaryDataset(
#                 data_path=self.data_path,
#                 split='val', 
#                 transform=val_transforms,
#                 image_folder=self.image_folder,
#                 command_folder=self.command_folder,
#                 inter_skip_frame=self.inter_skip_frame,
#                 num_in_frame=self.num_in_frame,
#                 num_out_frame=self.num_out_frame,    
#             )

#         elif self.data_type == 'binary-multi':
#             self.train_dataset = BEVMultistepBinaryDataset(
#                 data_path=self.data_path,
#                 split='train', 
#                 transform=train_transforms,
#                 image_folder=self.image_folder,
#                 command_folder=self.command_folder,
#                 inter_skip_frame=self.inter_skip_frame,
#                 intra_skip_frame=self.intra_skip_frame,
#                 num_in_frame=self.train_num_in_frame,
#                 num_out_frame=self.train_num_out_frame,
#                 stride = self.stride,    
#             )
        
#             self.val_dataset = BEVMultistepBinaryDataset(
#                 data_path=self.data_path,
#                 split='val', 
#                 transform=val_transforms,
#                 image_folder=self.image_folder,
#                 command_folder=self.command_folder,
#                 inter_skip_frame=self.inter_skip_frame,
#                 intra_skip_frame=self.intra_skip_frame,
#                 num_in_frame=self.val_num_in_frame,
#                 num_out_frame=self.val_num_out_frame,
#                 stride = self.stride,   
#             )

#             self.test_dataset = BEVMultistepBinaryDataset(
#                 data_path=self.test['data_path'],
#                 split='test', 
#                 transform=val_transforms,
#                 image_folder=self.test['image_folder'],
#                 command_folder=self.command_folder,
#                 inter_skip_frame=self.test['inter_skip_frame'],
#                 intra_skip_frame=self.test['intra_skip_frame'],
#                 num_in_frame=self.test['num_in'],
#                 num_out_frame=self.test['num_out'],
#                 stride = self.test['sliding_window_stride'],   
#             )

#         else:
#             self.train_dataset = BEVRgbDataset(
#                 self.data_path,
#                 split='train',
#                 image_folder=self.image_folder,
#                 skip_frame=self.inter_skip_frame,
#                 transform=train_transforms,
#                 zero_out_red_channel=self.zero_out_red_channel)

#             self.val_dataset = BEVRgbDataset(
#                 self.data_path,
#                 split='val',
#                 image_folder=self.image_folder,
#                 skip_frame=self.inter_skip_frame,
#                 transform=val_transforms,
#                 zero_out_red_channel=self.zero_out_red_channel,
#                 data_type=self.data_type
#             )
        
        
#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.train_batch_size,
#             num_workers=self.num_workers,
#             drop_last=True,
#             shuffle=True,
#             pin_memory=self.pin_memory,
#         )

#     def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.val_batch_size,
#             num_workers=self.num_workers,
#             drop_last=True,
#             shuffle=False,
#             pin_memory=self.pin_memory,
#         )
    
#     def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.test['batch_size'],
#             num_workers= 1, #self.num_workers,
#             drop_last=True,
#             shuffle=True,
#             pin_memory=self.pin_memory,
#         )




