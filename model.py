import torch
from torch import nn
from collections import OrderedDict

from models import fusion_network
from models.sanet import sanet, densenet
from models.swin.Swin_Unetr import SwinUNETR
from models.tabnet.tab_network import TabNet
from models.tabtransformer.ft_transformer import FTTransformer
from torchsummary import summary
from Params import *

def generate_model(opt):
    assert opt.model_type in ['tabnet', 'tabtransformer', 'sanet', 'swin', 'fusionet']

    if opt.model_type == 'tabnet':

        model = TabNet(input_dim = NUM_TAB_FEATURES,
                       output_dim = opt.n_classes,
                       cat_idxs = [],
                       cat_dims = [],
                       cat_emb_dim=1,
                       n_steps = 3,
                       gamma = 0.8,
                       n_a = 2,
                       n_d = 64, # Embedding Size
                       virtual_batch_size=32,
                       mask_type='entmax',
                      )

    elif opt.model_type == 'tabtransformer':

        model = FTTransformer(
            categories = [],      # tuple containing the number of unique values within each category
            num_continuous = NUM_CONTINUOUS,     # number of continuous values
            dim = 32,                           # dimension, paper set at 32
            dim_out = 1,                        # binary prediction, but could be anything
            depth = 3,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.3,                 # post-attention dropout
            ff_dropout = 0.3                    # feed forward dropout
        )

    elif opt.model_type == 'sanet':

        model = sanet.SANet(
            num_classes=opt.n_classes,
            sample_size=opt.input_D,
            sample_duration=opt.input_W)

    elif opt.model_type == 'swin':

        model = SwinUNETR(in_channels=opt.in_channels,
                  img_size=(opt.input_D, opt.input_H, opt.input_W),
                  batch_size=opt.batch_size,
                  feature_size=NUM_SWIN_FEATURES,
                  drop_rate=SWIN_DROPOUT)

    elif opt.model_type == 'fusionet':
        assert opt.model_fusion_type in ['early', 'late', 'tabnet_att', 'vit_transformer', 
        'tab_transformer', 'multi_modal_transformer']

        if opt.model_fusion_type == 'early':

            model = fusion_network.EarlyFusionModel(
               input_dim = 64, # size of output of biattention model
               output_dim = opt.n_classes,
               num_classes=opt.n_classes,
               sample_size=opt.input_D,
               sample_duration=opt.input_W,           
               cat_emb_dim=1,
               n_steps = 6,
               n_a = 10,
               n_d = 64,
               virtual_batch_size=16,
               mask_type='entmax',
               v_dim = 1024,
               num_hid = 64, # batch_size * v_dim # this is the dim after biattention
               alpha = 3
              )

        elif opt.model_fusion_type == 'late':

            model = fusion_network.LateFusionModel(
               input_dim = 2,
               output_dim = opt.n_classes,
               num_classes=opt.n_classes,
               sample_size=opt.input_D,
               sample_duration=opt.input_W,      
               cat_emb_dim=1,
               n_steps = 6,
               n_a = 10,
               n_d = 64,
               virtual_batch_size=16,
               mask_type='entmax',
               v_dim = 1024,
               num_hid = 64, # batch_size * v_dim # this is the dim after biattention
               alpha = 3
              )

        elif opt.model_fusion_type == 'tabnet_att':

            # Unimodal classifiers: TabNet + SANet
			# Fusion: BAN attention + TabNet 
            model = fusion_network.TabNetBANFusionModel(
               input_dim = 64, # size of output of BAN model
               output_dim = opt.n_classes,
               num_classes=opt.n_classes,
               sample_size=opt.input_D,
               sample_duration=opt.input_W,               
               cat_emb_dim=1,
               n_steps = 6,
               n_a = 10,
               n_d = 64,
               virtual_batch_size=16,
               mask_type='entmax',
               v_dim = 1024,
               num_hid = 64, # batch_size * v_dim # this is the dim after BAN
               alpha = 1
              )

        elif opt.model_fusion_type == 'vit_transformer':

            # Unimodal classifiers: TabNet + SANet
			# Fusion: ViT Transformer encoder
            model = fusion_network.ViTTransformerFusionModel(
               input_dim = 64, # size of output of biattention model
               output_dim = opt.n_classes,
               num_classes=opt.n_classes,
               sample_size=opt.input_D,
               sample_duration=opt.input_W,               
               cat_emb_dim=1,
               n_steps = 6,
               n_a = 10,
               n_d = 64,
               virtual_batch_size=16,
               mask_type='entmax',
               v_dim = 1024,
               num_hid = 64, # this is the dim after biattention
               alpha = 3
              )

        elif opt.model_fusion_type == 'multi_modal_transformer':

            # Unimodal transformers: TabNet + Swin UNETR
			# Fusion: TabTransformer encoder
            model = fusion_network.MultiModalTransformerModel(
               input_dim = 64, # size of output of biattention model
               output_dim = opt.n_classes,
               num_classes = opt.n_classes,
               input_D = opt.input_D,
               input_H = opt.input_H,
               input_W = opt.input_W,
               batch_size = opt.batch_size,
               cat_emb_dim=1,
               n_steps = 6,
               n_a = 10,
               n_d = 64,
               virtual_batch_size=16,
               mask_type='entmax',
               v_dim = 1024,
               num_hid = 64, # batch_size * v_dim # this is the dim after biattention
               alpha = 3
              )

        elif opt.model_fusion_type == 'tab_transformer':

            model = fusion_network.TabTransformerFusionModel(
               input_dim = 64, # size of output of biattention model
               output_dim = opt.n_classes,
               num_classes=opt.n_classes,
               sample_size=opt.input_D,
               sample_duration=opt.input_W,               
               cat_emb_dim=1,
               n_steps = 6,
               n_a = 10,
               n_d = 64,
               virtual_batch_size=16,
               mask_type='entmax',
               v_dim = 1024,
               num_hid = 64, # batch_size * v_dim # this is the dim after biattention
               alpha = 3
              )

    if not opt.no_cuda and opt.model_type == 'sanet':
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
        model = model.cuda()
        net_dict = model.state_dict()

    # Load pretrained model:
    if opt.pretrain_path:

        if opt.model_type == 'swin':
            print ('loading pretrained model {}'.format(opt.pretrain_path))
            model_weights = torch.load(opt.pretrain_path)
            model.load_from(model_weights)

        else:
            print ('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in  net_dict and v.size() == net_dict[k].size() }

            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict, strict=False)
            print('Loaded pretrained weights for {}'.format(opt.pretrain_path)) 

    # load pretrained uni-modality classifiers for image and tabuler data for multimodal fusion net
    # If model is multimodal and we want to use pretrained uni-modality classifiers:
    # 1. set opt.pretrain = '' 
    # 2. set opt.unimodality_pretrain = True

    if opt.model_type == 'fusionet' and opt.unimodality_pretrain:

        pretrained_dict = {}
        #Load pre-trained weights for tabular model and image model:

        if opt.img_pretrain_path:
            img_weight_dir = opt.img_pretrain_path
            print ('loading pretrained image model {}'.format(img_weight_dir))

            img_checkpoint = torch.load(img_weight_dir)
            for key in img_checkpoint['state_dict'].keys():
                pretrained_dict[key.replace("module", "module.img_emb")] = img_checkpoint['state_dict'][key]

        if opt.tab_pretrain_path:
            tab_weight_dir = opt.tab_pretrain_path
            print ('loading pretrained TabNet tabular model {}'.format(tab_weight_dir))
            
            tab_checkpoint = torch.load(tab_weight_dir)
            for key in tab_checkpoint['state_dict'].keys():
                pretrained_dict[key.replace("module.tabnet", "module.tab_emb")] = tab_checkpoint['state_dict'][key]

        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict, strict=False)

    if opt.model_freeze:
        for name, layer in model.named_parameters():
           for layer_name in opt.layers_to_freeze:
               if name.find(layer_name) >= 0:
                   layer.requires_grad = False

    new_parameters = [] 
    for pname, p in model.named_parameters():
        for layer_name in opt.new_layer_names:
            if pname.find(layer_name) >= 0:
                new_parameters.append(p)
                break

    new_parameters_id = list(map(id, new_parameters))
    base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    parameters = {'base_parameters': base_parameters, 
                  'new_parameters': new_parameters}
    if opt.pretrain_path:
        return model, parameters
    else: 
        return model, model.parameters()
