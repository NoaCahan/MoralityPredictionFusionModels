import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from models.tabnet import sparsemax
from models.tabnet.utils import define_device
from models.tabnet.utils import create_explain_matrix

from models.tabnet.attention import BiAttention
from models.tabnet.classifier import SimpleClassifier
from models.tabnet.fc import FCNet
from models.tabnet.bc import BCNet
from models.tabnet.tab_network import TabNet, BANmodel, EmbeddingGenerator
from models.sanet.sanet import SANet
from models.swin.Swin_Unetr import * #SwinUNETR
from Params import *

from models.transformer.backbone import Joiner
from models.transformer.position_encoding import build_position_encoding
from models.transformer.transformer import FusionTransformerEncoder, Transformer
from models.tabtransformer.ft_transformer import FTTransformer
from torch import nn
from setting import parse_opts
from models.transformer.utils import (NestedTensor, nested_tensor_from_tensor_list,
                       _onnx_nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

class EarlyFusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        sample_duration,
        sample_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 4,
        op='',
        num_classes=2):
        '''
            Early fusion of TabNet and SANet with TabNet model:
            1. Extract the image representation features and the Tabular embeddings.
            2. Aggregate them using Bilinear Attention and TabNet.
        '''
        super(EarlyFusionModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        
        # Tabular and Image single modality models
        self.img_emb = SANet(num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)


        self.embedder = EmbeddingGenerator(self.input_dim, self.cat_dims, 
                                           self.cat_idxs, self.cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        # Defining device
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))
        self.virtual_batch_size = virtual_batch_size

        self.v_att = BiAttention(v_dim, num_hid, num_hid, alpha)
        self.use_counter = False
        priotize_using_counter = None
        num_ans_candidates = 1
        
        # Optional module: counter for BAN
        use_counter = self.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None

        # init BAN residual network
        b_net = []
        q_prj = []
        c_prj = []
        for i in range(self.glimpse):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=3))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    
        # init classifier
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans_candidates, .5)

        self.banmodel = BANmodel(self.v_att, b_net, q_prj, c_prj, 
                                  classifier, counter, self.glimpse)
        self.tabnet_emb = TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)

        #Conv layer for image and tabular data
        self.batch_norm = torch.nn.BatchNorm1d(1)
        conv_img = torch.nn.Conv1d(
            1, 
            1024, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_img = torch.nn.utils.weight_norm(conv_img, dim=None)

        conv_tab = torch.nn.Conv1d(
            1, 
            self.input_dim, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_tab = torch.nn.utils.weight_norm(conv_tab, dim=None)


    def forward(self, img, ehr):
        _, i_emb = self.img_emb(img)
        _, _, e_emb = self.embedder(ehr)

        img = torch.unsqueeze(i_emb, 1)
        img = self.batch_norm(img)
        img = torch.nn.functional.relu(self.conv_img(img))

        ehr = torch.unsqueeze(e_emb, 1)
        ehr = self.batch_norm(ehr)
        ehr = torch.nn.functional.relu(self.conv_tab(ehr))

        att = self.banmodel(img, ehr)
        return self.tabnet_emb(att)

    def forward_masks(self, img, ehr):
        return self.tab_emb.forward_masks(ehr)

class LateFusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        sample_duration,
        sample_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 4,
        op='',
        num_classes=2):
        '''
            Extract prediction probabilities from each single modality and concat into TabNet.

            Late fusion of TabNet and SANet with TabNet model:
            1. Extract from each single modality theprediction probabilities.
            2. Aggregate them using TabNet.
        '''
        super(LateFusionModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        
        # Tabular and Image single modality models
        self.img_prob = SANet(num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)

        self.tab_prob = TabNet(input_dim = NUM_TAB_FEATURES,
               output_dim = self.num_classes,
               cat_idxs = [],
               cat_dims = [],
               cat_emb_dim=1,
               n_steps = 3,
               gamma = 0.8,
               n_a = 2,
               n_d = 64,
               virtual_batch_size=32,
               mask_type='entmax',
              )
        
        self.post_embed_dim = self.input_dim
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))
        self.virtual_batch_size = virtual_batch_size

        self.tabnet_emb = TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)

    def forward(self, img, ehr):
        i_prob, i_emb = self.img_prob(img)
        e_prob, _, e_emb = self.tab_prob(ehr)

        att = torch.cat((i_prob, e_prob), 0)
        return self.tabnet_emb(att)

    def forward_masks(self, img, ehr):
        return self.tab_emb.forward_masks(ehr)

class TabNetBANFusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        sample_duration,
        sample_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 3,
        op='',
        num_classes=2):
        '''
            Fusion of TabNet and SANet with Bilinear Attention BAN model:
            1. Extract from each single modality embeddings.
            2. Apply BAN model on extracted embeddings.
            3. TabNet on the attentive shared vector (BAN output)
        '''
        super(TabNetBANFusionModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        # Defining device
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))

        # Tabular and Image single modality models
        self.img_emb = SANet(num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)

        self.tab_emb = TabNet(input_dim = NUM_TAB_FEATURES,
               output_dim = self.num_classes,
               cat_idxs = [],
               cat_dims = [],
               cat_emb_dim=1,
               n_steps = 3,
               gamma = 0.8,
               n_a = 2,
               n_d = 64,
               virtual_batch_size=32,
               mask_type='entmax',
              ).to(self.device)
 
        self.post_embed_dim = self.input_dim

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size

        self.v_att = BiAttention(v_dim, num_hid, num_hid, alpha)
        self.use_counter = False
        priotize_using_counter = None
        num_ans_candidates = 1
        
        # Optional module: counter for BAN
        use_counter = self.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None

        # init BAN residual attention network
        b_net = []
        q_prj = []
        c_prj = []
        for i in range(self.glimpse):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=3))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    
        # init classifier
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans_candidates, .5)

        self.banmodel = BANmodel(self.v_att, b_net, q_prj, c_prj, 
                                  classifier, counter, self.glimpse)
        self.tabnet_emb = TabNet(
            self.input_dim,
            self.output_dim,
            n_d=10,#self.n_d,
            n_a=64,#self.n_a,
            n_steps=6,#,self.n_steps,
            #gamma=self.gamma,
            cat_idxs=[],#self.cat_idxs,
            cat_dims=[],#self.cat_dims,
            cat_emb_dim=1,#self.cat_emb_dim,
            #n_independent=self.n_independent,
            #n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=16,#self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)

        #Conv layer for image and tabular data
        self.batch_norm = torch.nn.BatchNorm1d(1)
        conv_img = torch.nn.Conv1d(
            1, 
            1024, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_img = torch.nn.utils.weight_norm(conv_img, dim=None)

        conv_tab = torch.nn.Conv1d(
            1, 
            self.input_dim, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_tab = torch.nn.utils.weight_norm(conv_tab, dim=None)


    def forward(self, img, ehr):
        _, i_emb = self.img_emb(img)
        _, _, e_emb = self.tab_emb(ehr)

        img = torch.unsqueeze(i_emb, 1)
        img = self.batch_norm(img)
        img = torch.nn.functional.relu(self.conv_img(img))

        ehr = torch.unsqueeze(e_emb, 1)
        ehr = self.batch_norm(ehr)
        ehr = torch.nn.functional.relu(self.conv_tab(ehr))

        att = self.banmodel(img, ehr)
        return self.tabnet_emb(att)

    def forward_masks(self, img, ehr):
        return self.tab_emb.forward_masks(ehr)

class ViTTransformerFusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        sample_duration,
        sample_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 4,
        op='',
        num_classes=2):
        '''
            Combine TabNet and SANet with ViT transformer model:
            1. Extract from each single modality embeddings.
            2. Apply BAN model on extracted embeddings.
            3. Apply ViT transformer encoder model onn BAN output.
        '''
        super(ViTTransformerFusionModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # Fusion Transformer arguments
        args = parse_opts()
        self.num_queries = 1 # For object detection
        self.num_decoder_layers = args.num_decoder_layers
        self.num_att_channels = args.num_att_channels # 64
        self.hidden_dim = args.hidden_dim # 256
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.num_att_channels, self.hidden_dim, kernel_size=1)

        # Tabular and Image single modality models
        self.img_emb = SANet(num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)

        self.tab_emb = TabNet(input_dim = NUM_TAB_FEATURES,
               output_dim = self.num_classes,
               cat_idxs = [],
               cat_dims = [],
               cat_emb_dim=1,
               n_steps = 3,
               gamma = 0.8,
               n_a = 2,
               n_d = 64,
               virtual_batch_size=32,
               mask_type='entmax',
              )
 
        self.post_embed_dim = self.input_dim
        # Defining device
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size

        self.v_att = BiAttention(v_dim, num_hid, num_hid, alpha)
        self.use_counter = False
        priotize_using_counter = None
        num_ans_candidates = 1
        
        # Optional module: counter for BAN
        use_counter = self.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None

        # init BAN residual network
        b_net = []
        q_prj = []
        c_prj = []
        for i in range(self.glimpse):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=3))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    
        # init classifier
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans_candidates, .5)

        self.banmodel = BANmodel(self.v_att, b_net, q_prj, c_prj, 
                                  classifier, counter, self.glimpse)

        # ViT Transformer encoder
        self.position_embedding = build_position_encoding()
        self.joiner = Joiner(self.position_embedding)
        self.encoder = FusionTransformerEncoder(d_model = self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim*64, self.hidden_dim)
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)

        self.batch_norm = torch.nn.BatchNorm1d(1)
        conv_img = torch.nn.Conv1d(
            1, 
            1024, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_img = torch.nn.utils.weight_norm(conv_img, dim=None)

        conv_tab = torch.nn.Conv1d(
            1, 
            self.input_dim, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_tab = torch.nn.utils.weight_norm(conv_tab, dim=None)


    def forward(self, img, ehr):

        _, i_emb = self.img_emb(img)
        _, _, e_emb = self.tab_emb(ehr)

        img = torch.unsqueeze(i_emb, 1)
        img = self.batch_norm(img)
        img = torch.nn.functional.relu(self.conv_img(img))

        ehr = torch.unsqueeze(e_emb, 1)
        ehr = self.batch_norm(ehr)
        ehr = torch.nn.functional.relu(self.conv_tab(ehr))

        att = self.banmodel(img, ehr)
        att = torch.unsqueeze(att, 1)
        att = torch.unsqueeze(att, 1)

        src_stack = []
        mask_stack = []
        pos_stack = []
        for batch_id in range(att.shape[0]):
            nest_att = nested_tensor_from_tensor_list([att[batch_id]])
            features, pos = self.joiner(nest_att)
            src, mask = features.decompose()
            assert mask is not None

            src_stack.append(src)
            mask_stack.append(mask)
            pos_stack.append(pos)
        src = torch.stack(src_stack).squeeze(1)
        mask = torch.stack(mask_stack).squeeze(1)
        pos = torch.stack(pos_stack).squeeze(1)

        hs  = self.encoder(self.input_proj(src), mask, self.query_embed.weight, pos)
        outputs = self.fc(hs.flatten(1))
        outputs_class = self.class_embed(outputs)
        
        return outputs_class, None, None


    def forward_masks(self, img, ehr):
        return self.tab_emb.forward_masks(ehr)

class TabTransformerWithAttFusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        sample_duration,
        sample_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 3,
        op='',
        num_classes=2):
        '''
            Combine TabNet with BiAttention BAN model:
            1. Extract from each single modality embeddings.
            2. Apply BAN model - BiAttention on extracted embeddings.
            3. TabNet on the attentive shared vector (BiAttention output)
        '''
        super(TabTransformerWithAttFusionModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        
        # Tabular and Image single modality models
        self.img_emb = SANet(num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)

        self.tab_emb = TabNet(input_dim = NUM_TAB_FEATURES,
               output_dim = self.num_classes,
               cat_idxs = [],
               cat_dims = [],
               cat_emb_dim=1,
               n_steps = 3,
               gamma = 0.8,
               n_a = 2,
               n_d = 64,
               virtual_batch_size=32,
               mask_type='entmax',
              )
 
        self.post_embed_dim = self.input_dim
        # Defining device
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size

        self.v_att = BiAttention(v_dim, num_hid, num_hid, alpha)
        self.use_counter = False
        priotize_using_counter = None
        num_ans_candidates = 1
        
        # Optional module: counter for BAN
        use_counter = self.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None

        # init BAN residual network
        b_net = []
        q_prj = []
        c_prj = []
        for i in range(self.glimpse):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=3))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    
        # init classifier
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans_candidates, .5)

        self.banmodel = BANmodel(self.v_att, b_net, q_prj, c_prj, 
                                  classifier, counter, self.glimpse)
        self.encoder = FTTransformer(            
		    categories = [],      # tuple containing the number of unique values within each category
            num_continuous = 64,#NUM_CONTINUOUS,     # number of continuous values
            dim = 32, # was 32                           # dimension, paper set at 32
            dim_out = 1,                        # binary prediction, but could be anything
            depth = 6,#6,                          # depth, paper recommended 6
            heads = 8,#8        ,                  # heads, paper recommends 8
            attn_dropout = 0.4,  # was 0.4               # post-attention dropout
            ff_dropout = 0.5          # was 0.5          # feed forward dropout
        )

        #Conv layer for image and tabular data
        self.batch_norm = torch.nn.BatchNorm1d(1)
        conv_img = torch.nn.Conv1d(
            1, 
            1024, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_img = torch.nn.utils.weight_norm(conv_img, dim=None)

        conv_tab = torch.nn.Conv1d(
            1, 
            self.input_dim, 
            kernel_size=3, 
            stride = 1, 
            padding=1,  
            groups=1, 
            bias=False)
        self.conv_tab = torch.nn.utils.weight_norm(conv_tab, dim=None)


    def forward(self, img, ehr):
        _, i_emb = self.img_emb(img)
        _, _, e_emb = self.tab_emb(ehr)

        img = torch.unsqueeze(i_emb, 1)
        img = self.batch_norm(img)
        img = torch.nn.functional.relu(self.conv_img(img))

        ehr = torch.unsqueeze(e_emb, 1)
        ehr = self.batch_norm(ehr)
        ehr = torch.nn.functional.relu(self.conv_tab(ehr))

        att = self.banmodel(img, ehr)
        return self.encoder(att)

class TabTransformerFusionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        sample_duration,
        sample_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 4,
        op='',
        num_classes=2):
        '''
            Combine TabNet and SANet with TabTransformer model:
            1. Extract from each single modality embeddings.
            2. Apply TabTransformer encoder model part.
        '''
        super(TabTransformerFusionModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # Fusion Transformer arguments
        args = parse_opts()
        self.num_queries = 1 # For object detection
        self.num_decoder_layers = args.num_decoder_layers
        self.num_att_channels = args.num_att_channels # 64
        self.hidden_dim = args.hidden_dim # 256
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.num_att_channels, self.hidden_dim, kernel_size=1)

        # Tabular and Image single modality models
        self.img_emb = SANet(num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)

        self.tab_emb = TabNet(input_dim = NUM_TAB_FEATURES,
               output_dim = self.num_classes,
               cat_idxs = [],
               cat_dims = [],
               cat_emb_dim=1,
               n_steps = 3,
               gamma = 0.8,
               n_a = 2,
               n_d = 64,
               virtual_batch_size=32,
               mask_type='entmax',
              )
 
        self.post_embed_dim = self.input_dim
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size

        self.encoder = FTTransformer(            
		    categories = [],      # tuple containing the number of unique values within each category
            num_continuous = 64+1024,#NUM_CONTINUOUS,     # number of continuous values
            dim = 32,#32,                           # dimension, paper set at 32
            dim_out = 1,                        # binary prediction, but could be anything
            depth = 18,#6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.7,                 # post-attention dropout
            ff_dropout = 0.9                   # feed forward dropout
        )

    def forward(self, img, ehr):

        _, i_emb = self.img_emb(img)
        _, _, e_emb = self.tab_emb(ehr)

        att = torch.cat((e_emb, i_emb),dim=1)

        return self.encoder(att)

    def forward_masks(self, img, ehr):
        return self.tab_emb.forward_masks(ehr)

class MultiModalTransformerModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        v_dim, 
        num_hid,
        input_D,
        input_H,
        input_W,
        batch_size,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax", 
        alpha = 4,
        op='',
        num_classes=2):
        '''
            Combine TabNet and Swin with TabTransformer model:
            1. Extract from each single modality embeddings.
            2. Apply TabTransformer encoder model part.
        '''
        super(MultiModalTransformerModel, self).__init__()
        
        self.cat_idxs = []
        self.cat_dims = []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.glimpse = alpha
        self.momentum = momentum
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Fusion Transformer arguments
        args = parse_opts()
        self.num_queries = 1 # For object detection
        self.num_decoder_layers = args.num_decoder_layers
        self.num_att_channels = args.num_att_channels # 64
        self.hidden_dim = args.hidden_dim # 256
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.num_att_channels, self.hidden_dim, kernel_size=1)

        # Tabular and Image single modality models
        self.img_emb = SwinUNETR(in_channels=NUM_CHANNELS,
                  batch_size = batch_size,
                  img_size=(self.input_D, self.input_H, self.input_W),
                  feature_size=NUM_SWIN_FEATURES,
                  drop_rate=SWIN_DROPOUT)

        self.tab_emb = TabNet(input_dim = NUM_TAB_FEATURES,
               output_dim = self.num_classes,
               cat_idxs = [],
               cat_dims = [],
               cat_emb_dim=1,
               n_steps = 3,
               gamma = 0.8,
               n_a = 2,
               n_d = 64,
               virtual_batch_size=32,
               mask_type='entmax',
              )
 
        self.post_embed_dim = self.input_dim
        # Defining device
        self.device_name = 'auto'
        self.device = torch.device(define_device(self.device_name))

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size

        self.encoder = FTTransformer(            
		    categories = [],      # tuple containing the number of unique values within each category
            num_continuous = 64+1024,           # the concatenated embedding size from the two modalities 
            dim = 32,                           # dimension, paper set at 32
            dim_out = 1,                        # binary prediction, but could be anything
            depth = 6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.3,                 # post-attention dropout
            ff_dropout = 0.3                    # feed forward dropout
        )

    def forward(self, img, ehr):
        _, i_emb = self.img_emb(img)
        _, _, e_emb = self.tab_emb(ehr)
        if len(i_emb.shape) == 1:
            i_emb = i_emb.reshape(1,i_emb.shape[0])

        att = torch.cat((e_emb, i_emb),dim=1)
        return self.encoder(att)

    def forward_masks(self, img, ehr):
        return self.tab_emb.forward_masks(ehr)