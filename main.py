import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import argparse
import load_data
from datetime import datetime
from matplotlib.pylab import rcParams
from matplotlib.pyplot import text
from sklearn.metrics import f1_score, roc_curve, auc, recall_score, precision_score, average_precision_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import (
    SGD, Adam
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    Callback, 
    TensorBoard,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from models.densenet1d import build_dense_net
from training_utils import (
    get_one_cycle_lr_fn,
    get_one_cycle_momentum_fn
)
from momentum_scheduler import MomentumScheduler

MODEL_FOLDER = "saved_models"
RESULT_FOLDER = "results"
TB_FOLDER = "tblogs"

# Build and training 1-d dense net
def train_1d_densenet(
    model_name, X_train, X_valid, y_train, y_valid, sample_weights,
    input_shape, output_dim, 
    lr=0.01, dpt=0.5, mid_dpt=0.0, cvf=128, cvs=7, mid_cvs=3,
    batch_size=128, blocks = [3, 6, 12, 8], growth_rate=32,
    epochs = 1000, optimizer="SGD", use_l2=1, l2_val=0.001,
    earlystop=1, patience=5, RLR=0, 
    summary=True, use_tb=0
):

    model = build_dense_net(
        blocks, output_dim, input_shape=input_shape, 
        growth_rate=growth_rate,
        lr=lr, dpt=dpt, mid_dpt=mid_dpt,
        cvf=cvf, cvs=cvs,
        mid_cvs=mid_cvs,
        use_l2=use_l2, l2_val=l2_val
    )

    if summary:
        model.summary()

    model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath = f"{MODEL_FOLDER}/{model_name}-best.h5", verbose=1, 
                                       save_best_only=True, mode="min", save_weights_only=False)

    callbacks = [model_checkpoint]

    if earlystop == 1:
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience, mode="auto")
        callbacks.append(early_stopping)

    if RLR == 1:
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, min_delta=1e-4)
        callbacks.append(reduce_lr)

    if use_tb == 1:
        if not os.path.exists(f"{TB_FOLDER}/{model_name}"):
            os.makedirs(f"{TB_FOLDER}/{model_name}")

        tb = TensorBoard(log_dir=f"./{TB_FOLDER}/{model_name}", 
            histogram_freq=0, 
            write_graph=False, 
            write_grads=False, 
            write_images=False, 
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None, 
            embeddings_data=None, 
            update_freq="epoch"
        )

        callbacks.append(tb)

    if optimizer.lower() == "sgd":
        optimizer = SGD(learning_rate=lr, decay=1e-6)
    elif optimizer.lower() == "1cycle":
        lr_func  = get_one_cycle_lr_fn(epochs, lr, np.floor(epochs*0.1))
        mom_func = get_one_cycle_momentum_fn(epochs, lr, np.floor(epochs*0.1))

        lr_scheduler = LearningRateScheduler(lr_func)
        mom_scheduler = MomentumScheduler(mom_func)

        optimizer = SGD(learning_rate=lr_func(0), momentum=mom_func(0))

        callbacks.append(lr_scheduler)
        callbacks.append(mom_scheduler)
    else:
        optimizer = Adam(learning_rate=lr)
    
    if output_dim > 1:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy'])
    else:
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['binary_accuracy'])

    history = model.fit(X_train, y_train,
                    epochs=epochs,
                    sample_weight=sample_weights,
                    batch_size=batch_size,
                    validation_data=(X_valid, y_valid),
                    callbacks=callbacks,
                    verbose=1)

    return history

# Test model
def test_1d_densenet(model_name, test_X, test_y):
    clf = load_model(f"{MODEL_FOLDER}/{model_name}-best.h5")
    
    proba = clf.predict(test_X, verbose=0) 
    
    y_true = test_y.argmax(axis=-1)
    y_pred_class = proba.argmax(axis=-1)

    cm = confusion_matrix(y_true, y_pred_class)
    rpt = classification_report(y_true, y_pred_class)

    rpt += "\n" + str(cm)
    
    return rpt


parser = argparse.ArgumentParser (description='NBS')

parser.add_argument('-gpu', metavar='int', type=str, default='6', help='gpu index, using which GPU')
parser.add_argument('-train', metavar='train', type=int, default='0', help='0:train; 1:load_pretrained')
parser.add_argument('-name', metavar='str', type=str, default='Experiment', help='Name to identify experiment results.')
parser.add_argument('-lr', metavar='lr', type=float, default='0.01', help='Learning rate.')
parser.add_argument('-dpt', metavar='dpt', type=float, default='0.5', help='Dropout')
parser.add_argument('-mid_dpt', metavar='mid_dpt', type=float, default='0.0', help='Dropout')
parser.add_argument('-cvf', metavar='cvf', type=int, default=128, help='First conv layer filters')
parser.add_argument('-cvs', metavar='cvs', type=int, default=7, help='First conv layer size')
parser.add_argument('-mid_cvs', metavar='mid_cvs', type=int, default=3, help='Inner conv layer size')
parser.add_argument('-batch_size', metavar='batch_size', type=int, default=128, help='batch_size')
parser.add_argument('-blocks', type=str, default="3,6,12,8")
parser.add_argument('-growthrate', type=int, default=32)
parser.add_argument('-epochs', type=int, default=1000)
parser.add_argument('-opt', type=str, default="SGD", choices=["SGD", "Adam", "1cycle"])
parser.add_argument('-scaling', type=str, default="standard")
parser.add_argument('-data', type=str, default="std", choices=["std", "big", "lsr"])
parser.add_argument('-dist_matrix', type=str, default="orig", choices=["orig", "precomp"])

# These are boolean variables, but we use 0 and 1 indicators rather than boolean
# flags because it's just easier to pass them from the manifest

parser.add_argument('-earlystop', type=int, default=0, choices=[0,1], help="Pass 1 to use early stopping")
parser.add_argument('-patience', type=int, default=5)
parser.add_argument('-RLR', type=int, default=0, choices=[0,1], help="Pass 1 to use the reduce learning rate callback")
parser.add_argument('-use_sw', type=int, default=0, choices=[0,1], help="Pass 1 to use NB weighting, 0 otherwise")
parser.add_argument('-use_l2', type=int, default=1, choices=[0,1], help="Pass 1 to use l2 regularization, 0 otherwise")
parser.add_argument('-use_tb', type=int, default=0, choices=[0,1], help="Pass 1 to use tensorboard, 0 otherwise")
parser.add_argument('-eval_set', type=str, default="valid", choices=["valid", "test"], help="Pass 'valid' or 'test' to specify the evaluation set")

# Text file name to dump results

parser.add_argument('-resfilename', type=str, default="results")

if __name__== "__main__": 
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model_name = args.name

    X_train, X_eval, y_train, y_eval, sample_weights = load_data.get_densenet_samples(
        scale=args.scaling, data=args.data, mode=args.eval_set,
        calc_scores=bool(args.use_sw), dist_matrix=args.dist_matrix
    )

    if args.train==0:
        history = train_1d_densenet(
            model_name,
            X_train, X_eval, y_train, y_eval, sample_weights, 
            input_shape=X_train.shape[1:], output_dim= y_train.shape[1],
            lr=args.lr, dpt=args.dpt, mid_dpt=args.mid_dpt, cvf=args.cvf, cvs=args.cvs, mid_cvs=args.mid_cvs,
            batch_size=args.batch_size, blocks=[int(x) for x in args.blocks.split(',')], growth_rate=args.growthrate,
            epochs=args.epochs, earlystop=args.earlystop, patience=args.patience, RLR=args.RLR,
            use_l2=args.use_l2, optimizer=args.opt, use_tb=args.use_tb
        )

    log_str = f"Experiment name: {args.name}, finished training at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} UTC\nScaling:{args.scaling}\n"
    log_str += f"Hyperparams:\n\nLearning rate: {args.lr} Dropout: {args.dpt} Mid dropout: {args.mid_dpt}\n"
    log_str += f"Conv filters: {args.cvf} 1st Conv size: {args.cvs} Mid Conv Size:{args.mid_cvs} Epochs: {args.epochs}\n"
    log_str += f"Optimizer: {args.opt} L2 on?: {args.use_l2} Epochs: {args.epochs}\n"
    log_str += f"Batch size: {args.batch_size} Network blocks: {args.blocks}\n\n"

    log_str += f"Results on {args.eval_set}: \n\n"

    last_train_acc = history.history['categorical_accuracy'][-1]
    last_valid_acc = history.history['val_categorical_accuracy'][-1]
    number_of_training_epochs = len(history.history['categorical_accuracy'])

    log_str += f"Final training accuracy: {last_train_acc:.03f}, Final validation accuracy: {last_valid_acc:.03f}\n"
    log_str += f"Number of epochs trained: {number_of_training_epochs}\n\n"
    log_str += f"{args.eval_set} set result:\n\n"

    log_str += test_1d_densenet(model_name, X_eval, y_eval) + "\n\n"

    with open(f"./{RESULT_FOLDER}/{args.resfilename}.txt", "a+") as file:
        file.write(log_str)
