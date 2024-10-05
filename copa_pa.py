from multiprocessing import context
import os
import sys
import torch
import wandb
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir, Recorder, setup_seed, device

from models.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.adaptation import pa_adaptation
from models.transformer import Transformer
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args
tf.compat.v1.disable_eager_execution()

ROOT_PATH = os.path.join('/tmp/plot/exp_saved_data/CLIPCFC', args['exp_dir_name'])

def main():
    # set seed
    setup_seed(seed_id=args['seed'])

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    
    if args['test.mode'] == 'mdl':
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        trainsets = ['ilsvrc_2012']
    
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()

    # Initialize CLIP head
    if args['encoder.type'] == 'linear':
        feature_encoder = None
        prototype_encoder = None

    elif args['encoder.type'] == 'transformer':
        prototype_encoder = Transformer(num_blocks=args['transformer.n_blocks'], 
                                    num_dim=args['transformer.dim'], 
                                    is_simpleatten=True,
                                    init_mode='eye').to(device)
        
        feature_encoder = Transformer(num_blocks=args['transformer.n_blocks'],
                                      num_dim=args['transformer.dim'],
                                      is_simpleatten=True,
                                      init_mode='eye').to(device)
    else:
        raise ValueError("Unrecognized feature encoder type. Please choose from ['linear', 'transformer'].")

    # Results recorder
    recorder = Recorder(saveroot=ROOT_PATH, datasets=testsets,
                        key_wd_list=['train_losses', 'train_accs', 'val_losses', 'val_accs'])

    accs_names = ['NCC']
    train_var_accs = dict()
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(dataset)
            train_var_accs[dataset] = {name:[] for name in accs_names}
            var_accs[dataset] = {name: [] for name in accs_names}
            
            for i in tqdm(range(args['test.size'])):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    #support_images = sample['context_images']
                    #query_images = sample['target_images']
                    context_features = model.embed(sample['context_images'], is_pooling=False)
                    target_features = model.embed(sample['target_images'], is_pooling=False)
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']

                # optimize selection parameters and perform feature selection
                data_recorder = pa_adaptation(context_features, context_features, context_labels,
                                              target_features, target_labels,
                                              prototype_encoder, feature_encoder,
                                              dataset_name=dataset, max_iter=50)

                train_var_accs[dataset]['NCC'].append(data_recorder['train_accs'][-1])
                var_accs[dataset]['NCC'].append(data_recorder['val_accs'][-1])

                recorder.update_records(dataset, data_recorder)

            train_acc = np.array(train_var_accs[dataset]['NCC'])*100
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: train_acc {train_acc.mean():.2f}%; test_acc {dataset_acc.mean():.2f} +/- {(1.96*dataset_acc.std()) / np.sqrt(len(dataset_acc)):.2f}%")
    recorder.save(filename=args['experiment.name'])
    # Print nice results table
    print('results of {}'.format(args['experiment.name']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], 'pa', args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")

if __name__ == '__main__':
    main()