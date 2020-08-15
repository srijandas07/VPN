#------------------------------------------
# Model train file for VPN
# Created By Srijan Das and Saurav Sharma
#------------------------------------------

import os, sys
from pathlib import Path
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import multi_gpu_model

from multiprocessing import cpu_count
from NTU_gcnn_Loader import *
from embedding_gcnn_attention_model import *
from compute_adjacency import *
from utils import *
from main import generate_config, parse_args

def set_seed(value):
    np.random.seed(seed)

class CustomModelCheckpoint(Callback):
    def __init__(self, model_parallel, path):
        super(CustomModelCheckpoint, self).__init__()
        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')

def gcnn_model(args):
    A = compute_adjacency(args.dataset, args.alpha, args.beta)
    A = np.repeat(A, args.batch_size, axis=0)
    A = np.reshape(A, [args.batch_size, A.shape[1], A.shape[1]]) 
    graph_conv_filters = preprocess_adj_tensor_with_identity(A, args.sym_norm)

    return graph_conv_filters

def trainer(args):
    # set seed first
    set_seed(8)

    # define undirected graph for Input poses
    graph_conv_filters = gcnn_model(args)

    # create vpn model
    model = embed_model_spatio_temporal_gcnn(n_neuron, timesteps, args.num_nodes, args.num_features,
                                        graph_conv_filters.shape[1], graph_conv_filters.shape[2],
                                        args.num_filters, args.num_classes, args.n_dropout, args.protocol)

    # define loss and weightage to different loss components
    losses = {
	            "action_output": "categorical_crossentropy",
	            "embed_output": "mean_squared_error",
            }
    lossWeights = {"action_output": args.action_wt, "embed_output": args.embed_wt}

    # define optimizer
    optim = SGD(lr=args.lr, momentum=args.momentum)
    
    # define data generators
    train_generator = DataGenerator(args.paths, graph_conv_filters, args.timesteps, args.train_ds_name, args.num_classes, args.stack_size, batch_size=args.batch_size)
    val_generator = DataGenerator(args.paths, graph_conv_filters, args.timesteps, args.test_ds_name, args.num_classes, args.stack_size, batch_size=args.batch_size)

    # compile model
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optim, metrics=['accuracy'])

    # model loggers
    csvlogger = CSVLogger('_'.join([args.model_name, args.dataset,'.csv']))
    reduce_lr = ReduceLROnPlateau(monitor=args.monitor, factor = args.factor, patience = args.patience)

    # select model training method - multi or single GPU
    print(f'Training for {args.dataset} dataset starts!')
    if args.multi_gpu:
        parallel_model = multi_gpu_model(model, gpus=args.num_gpus)
        parallel_model.compile(loss=losses, loss_weights=lossWeights, optimizer=optim, metrics=['accuracy'])

        model.compile(loss=losses, loss_weights=lossWeights, optimizer=optim, metrics=['accuracy'])

        # create folder to save model checkpoints if not already exists
        Path(os.path.join(args.weights_loc+args.model_name)).mkdir(parents=True, exist_ok=True)

        model_checkpoint = CustomModelCheckpoint(model, os.path.join(args.weights_loc+args.model_name,'epoch_')) # Not sure whether it should be model or parallel_model

        parallel_model.fit_generator(generator=train_generator,
                            validation_data=val_generator,
                            use_multiprocessing=args.multi_proc,
                            epochs=args.epochs,
                            callbacks = [csvlogger, model_checkpoint],
                            workers=cpu_count() - 2)
    print(f'Training for {args.dataset} dataset is complete!')


if __name__ == '__main__':
    args = generate_config()
    train(args)
