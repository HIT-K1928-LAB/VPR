# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
import yaml
from typing import Dict, Any


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='VPR Framework Training and Evaluation')

    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--train', action='store_true', help='Run mode: train or evaluate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--silent', action='store_true', help='Disable verbose output')
    parser.add_argument('--compile', action='store_true', help='Compile the model using torch.compile()')
    parser.add_argument('--dev', action='store_true', help='Enable fast development run')
    parser.add_argument('--test', action='store_true', help='Run evaluation instead of training')
    parser.add_argument('--ckpt_path', type=str, help='Path to a checkpoint for evaluation')
    parser.add_argument('--display_theme', type=str, help='Theme for the console display')

    parser.add_argument('--train_set', type=str, help='Name of the training dataset')
    parser.add_argument('--train_sets', nargs='+', help='Names of the training datasets')
    parser.add_argument('--train_dataset_weights', nargs='+', help='Weights for train datasets, e.g. name=1.0 msls=0.7')
    parser.add_argument('--train_loader_mode', type=str, help='CombinedLoader mode: min_size, max_size_cycle, max_size')
    parser.add_argument('--msls_sample_mode', type=str, help='MSLS sampling mode: clique, cluster, random, recent')
    parser.add_argument('--msls_bucket_size_m', type=float, help='MSLS bucket size in meters')
    parser.add_argument('--msls_cluster_radius_m', type=float, help='MSLS cluster radius in meters')
    parser.add_argument('--sf_xl_sample_mode', type=str, help='SF_XL sampling mode: clique, cluster, random, recent')
    parser.add_argument('--sf_xl_bucket_size_m', type=float, help='SF_XL bucket size in meters')
    parser.add_argument('--sf_xl_cluster_radius_m', type=float, help='SF_XL cluster radius in meters')
    parser.add_argument('--generic_sample_mode', type=str, help='Fallback sampling mode for other datasets')
    parser.add_argument('--val_sets', nargs='+', help='Names of the validation datasets')
    parser.add_argument('--train_image_size', type=int, nargs=2, help='Training image size (height width)')
    parser.add_argument('--val_image_size', type=int, nargs=2, help='Validation image size (height width). Dafault is None (same as training size)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--img_per_place', type=int, help='Number of images per place')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')

    parser.add_argument('--backbone', type=str, help='Backbone model name')
    parser.add_argument('--aggregator', type=str, help='Aggregator model name')
    parser.add_argument('--loss_function', type=str, help='Loss function name')

    parser.add_argument('--optimizer', type=str, help='Optimizer name')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--warmup', type=int, help='Number of warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, help='Milestones for learning rate scheduler')
    parser.add_argument('--lr_mult', type=float, help='Learning rate multiplier for scheduler')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--accelerator', type=str, help='Lightning accelerator, e.g. gpu, cpu, auto')
    parser.add_argument('--devices', nargs='+', help='Lightning devices spec, e.g. 1 or 0 1 for multi-GPU')
    parser.add_argument('--strategy', type=str, help='Lightning distributed strategy, e.g. ddp')
    parser.add_argument('--num_nodes', type=int, help='Number of nodes to use')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("No config file provided. Using command-line arguments and default values.")
        config = {}

    config = update_config_with_args_and_defaults(config, args)
    return config


def update_config_with_args_and_defaults(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    default_config = {
        'seed': 42,
        'silent': False,
        'compile': False,
        'dev': False,
        'display_theme': "default",
        'train': True,
        'ckpt_path': None,
        'datamodule': {
            'train_set_name': "gsv-cities-light",
            'train_set_names': None,
            'train_dataset_weights': None,
            'train_loader_mode': "max_size_cycle",
            'msls_sampling': {
                'sample_mode': 'clique',
                'bucket_size_m': 25.0,
                'cluster_radius_m': 20.0,
            },
            'sf_xl_sampling': {
                'sample_mode': 'clique',
                'bucket_size_m': 25.0,
                'cluster_radius_m': 20.0,
            },
            'generic_sampling': {
                'sample_mode': 'clique',
                'bucket_size_m': 25.0,
                'cluster_radius_m': 20.0,
            },
            'cities': "all",
            'val_set_names': ["msls-val"],
            'train_image_size': [320, 320],
            'val_image_size': None,
            'batch_size': 60,
            'img_per_place': 4,
            'num_workers': 8,
        },
        'backbone': {
            'module': 'src.models.backbones',
            'class': 'ResNet',
            'params': {},
        },
        'aggregator': {
            'module': 'src.models.aggregators',
            'class': 'MixVPR',
            'params': {},
        },
        'loss_function': {
            'module': 'src.losses',
            'class': 'VPRLossFunction',
            'params': {},
        },
        'trainer': {
            'optimizer': "adamw",
            'lr': 0.0002,
            'wd': 0.001,
            'warmup': 0,
            'milestones': [10, 20, 30],
            'lr_mult': 0.1,
            'max_epochs': 40,
            'accelerator': 'gpu',
            'devices': 1,
            'strategy': None,
            'num_nodes': 1,
        },
    }

    def update_nested_dict(d, u):
        if not isinstance(d, dict):
            d = {}
        for k, v in u.items():
            if isinstance(v, dict):
                child = d.get(k)
                if not isinstance(child, dict):
                    child = {}
                d[k] = update_nested_dict(child, v)
            else:
                d[k] = v
        return d

    config = update_nested_dict(default_config, config)
    arg_dict = vars(args)

    if arg_dict['train_set'] is not None:
        config['datamodule']['train_set_name'] = arg_dict['train_set']
        config['datamodule']['train_set_names'] = [arg_dict['train_set']]
    if arg_dict.get('train_sets') is not None:
        config['datamodule']['train_set_names'] = arg_dict['train_sets']
        config['datamodule']['train_set_name'] = arg_dict['train_sets'][0]
    if arg_dict['train_dataset_weights'] is not None:
        parsed_weights = {}
        for item in arg_dict['train_dataset_weights']:
            if '=' not in item:
                raise ValueError("train_dataset_weights must be in name=value form.")
            name, value = item.split('=', 1)
            parsed_weights[name] = float(value)
        config['datamodule']['train_dataset_weights'] = parsed_weights
    if arg_dict['train_loader_mode'] is not None:
        config['datamodule']['train_loader_mode'] = arg_dict['train_loader_mode']
    if arg_dict['msls_sample_mode'] is not None:
        config['datamodule']['msls_sampling']['sample_mode'] = arg_dict['msls_sample_mode']
    if arg_dict['msls_bucket_size_m'] is not None:
        config['datamodule']['msls_sampling']['bucket_size_m'] = arg_dict['msls_bucket_size_m']
    if arg_dict['msls_cluster_radius_m'] is not None:
        config['datamodule']['msls_sampling']['cluster_radius_m'] = arg_dict['msls_cluster_radius_m']
    if arg_dict['sf_xl_sample_mode'] is not None:
        config['datamodule']['sf_xl_sampling']['sample_mode'] = arg_dict['sf_xl_sample_mode']
    if arg_dict['sf_xl_bucket_size_m'] is not None:
        config['datamodule']['sf_xl_sampling']['bucket_size_m'] = arg_dict['sf_xl_bucket_size_m']
    if arg_dict['sf_xl_cluster_radius_m'] is not None:
        config['datamodule']['sf_xl_sampling']['cluster_radius_m'] = arg_dict['sf_xl_cluster_radius_m']
    if arg_dict['generic_sample_mode'] is not None:
        config['datamodule']['generic_sampling']['sample_mode'] = arg_dict['generic_sample_mode']
    if arg_dict['val_sets'] is not None:
        config['datamodule']['val_set_names'] = arg_dict['val_sets']
    if arg_dict['train_image_size'] is not None:
        config['datamodule']['train_image_size'] = arg_dict['train_image_size']
    if arg_dict['val_image_size'] is not None:
        config['datamodule']['val_image_size'] = arg_dict['val_image_size']
    if arg_dict['batch_size'] is not None:
        config['datamodule']['batch_size'] = arg_dict['batch_size']
    if arg_dict['img_per_place'] is not None:
        config['datamodule']['img_per_place'] = arg_dict['img_per_place']
    if arg_dict['num_workers'] is not None:
        config['datamodule']['num_workers'] = arg_dict['num_workers']

    if arg_dict['backbone'] is not None:
        config['backbone']['class'] = arg_dict['backbone']
    if arg_dict['aggregator'] is not None:
        config['aggregator']['class'] = arg_dict['aggregator']
    if arg_dict['loss_function'] is not None:
        config['loss_function']['class'] = arg_dict['loss_function']

    if arg_dict['optimizer'] is not None:
        config['trainer']['optimizer'] = arg_dict['optimizer']
    if arg_dict['lr'] is not None:
        config['trainer']['lr'] = arg_dict['lr']
    if arg_dict['wd'] is not None:
        config['trainer']['wd'] = arg_dict['wd']
    if arg_dict['warmup'] is not None:
        config['trainer']['warmup'] = arg_dict['warmup']
    if arg_dict['milestones'] is not None:
        config['trainer']['milestones'] = arg_dict['milestones']
    if arg_dict['lr_mult'] is not None:
        config['trainer']['lr_mult'] = arg_dict['lr_mult']
    if arg_dict['max_epochs'] is not None:
        config['trainer']['max_epochs'] = arg_dict['max_epochs']
    if arg_dict['accelerator'] is not None:
        config['trainer']['accelerator'] = arg_dict['accelerator']
    if arg_dict['devices'] is not None:
        config['trainer']['devices'] = arg_dict['devices']
    if arg_dict['strategy'] is not None:
        config['trainer']['strategy'] = arg_dict['strategy']
    if arg_dict['num_nodes'] is not None:
        config['trainer']['num_nodes'] = arg_dict['num_nodes']

    if arg_dict['seed'] is not None:
        config['seed'] = arg_dict['seed']
    if arg_dict['silent']:
        config['silent'] = arg_dict['silent']
    if arg_dict['compile']:
        config['compile'] = arg_dict['compile']
    if arg_dict['display_theme'] is not None:
        config['display_theme'] = arg_dict['display_theme']
    if arg_dict['dev']:
        config['dev'] = arg_dict['dev']
    if arg_dict['train']:
        config['train'] = arg_dict['train']
    if arg_dict['test']:
        config['train'] = False
    if arg_dict['ckpt_path'] is not None:
        config['ckpt_path'] = arg_dict['ckpt_path']

    return config


if __name__ == "__main__":
    config = parse_args()
