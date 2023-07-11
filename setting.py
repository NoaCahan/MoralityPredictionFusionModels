'''
Configs for training & testing
Written by Whalechen
'''

import argparse
from Params import *

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=ROOT,
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--in_channels',
        default=NUM_CHANNELS,
        type=int,
        help="Number of classification channels"
    )
    parser.add_argument(
        '--n_classes',
        default=NUM_CLASSES,
        type=int,
        help='Number of input classes'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', 
        default=2, 
        type=int, 
        help= 'Batch Size')
    parser.add_argument(
        '--num_decoder_layers',
        default=1,
        type=int,
        help= 'number of encoder\decoder layers for transformer')
    parser.add_argument(
        '--hidden_dim',
        default=128,
        type=int,
        help= 'hidden dim for transformer')
    parser.add_argument(
        '--num_att_channels',
        default=1,
        type=int,
        help= 'number of channels after cross attention')
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        help='position embeddings for transformer')
    parser.add_argument(
        '--lambda_sparse',
        default= 1e-08,
        type=float,
        help= 'sets the sparsity loss for tabnet model')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=50,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=500,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
    default=128,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=128,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=128,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--pretrain_path',
        default ='',
        #'path to pretrained model',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--unimodality_pretrain',
        default=False,
        type=bool,
        help= 'Use pretrained single modality classifiers for joint model or not. This is for fusion models'
    )
    parser.add_argument(
        '--img_pretrain_path',
        default='path to pretrained image model checkpoint',
        type=str,
        help= 'If unimodality_pretrain = True -  the locations to the image pretrained model'
    )
    parser.add_argument(
        '--tab_pretrain_path',
        default='path to pretrained tabular model checkpoint',
        type=str,
        help= 'If unimodality_pretrain = True - the locations to the tabular pretrained model'
    )
    parser.add_argument(
        '--model_freeze',
        default=False,
        type=bool,
        help= 'Freeze model or part of model'
    )
    parser.add_argument(
        '--layers_to_freeze',
        default=['tab_emb', 'img_emb'],
        type=list,
        help='Layers to freeze if model_freeze = True')

    parser.add_argument(
        '--new_layer_names',
        default=['tab_emb', 'img_emb'],
        type=list,
        help='New layer names for pretrained backbone'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        default=[0],
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model_type',
        default='fusionet',
        type=str,
        help='(tabnet | tabtransformer | sanet | swin | fusionet')
    parser.add_argument(
        '--model_fusion_type',
        default='tabnet_att',
        type=str,
        help='Fusion model type (early | late | tabnet_att | vit_transformer | tab_transformer | multi_modal_transformer) ')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    args = parser.parse_args()
    if args.model_type != 'fusionet':
        args.save_folder = "./results/models/{}".format(args.model_type)
    else:
        args.save_folder = "./results/models/{}_{}".format(args.model_type, args.model_fusion_type)
    
    return args
