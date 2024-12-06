import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from feature import calc_cqt, calc_stft, save_feature, load_feature
from feature import calc_wav2vec
from metrics import calculate_eer, calculate_classifier_metrics
from sklearn.utils import shuffle

from model.lcnn import build_lcnn
from model.lcnn_lstm import build_lcnn_lstm
from model.resnet18 import build_resnet
# from model.resnet18_lstm import build_resnet_lstm

import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------------------------------------------------------------------------------------
# cache and data
cache = '__cache__'
if not os.path.exists(cache):
    os.makedirs(cache)

log = '__log__'
if not os.path.exists(log):
    os.makedirs(log)

job_id = os.getenv('SLURM_JOB_ID')

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for training')

    parser.add_argument('--model', '-m', type=str, default="lcnn")
    parser.add_argument('--feature', '-f',  type=str, default="stft")
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--datasize', '-s', type=int, default=-1, help='data size')
    parser.add_argument('--verbose', type=int, default=-0, help='verbose 0 (silent), 1 (detailed output)')
    parser.add_argument('--savedata', type=int, default=1, help='save your data as npz')

    args = parser.parse_args()
    return args
# ---------------------------------------------------------------------------------------------------------------------------------------

CRED = '\033[91m'
CEND = '\033[0m'
args = parse_arguments()
print(f'*********** Job ID: {job_id} ***********')
for k,v in args.__dict__.items(): print(f'> args: {CRED} {k} {CEND}, value: {CRED} {v} {CEND}')


# feature type
feature_type = args.feature
feature_xtract_map = {
    'cqt': calc_cqt,
    'stft': calc_stft,
    # 'wav2vec': calc_wav2vec
}
if feature_type not in feature_xtract_map:
    raise ValueError(f'feature type "{feature_type}" not exist!')

# model type
model_type = args.model
model_build_map = {
    'lcnn': build_lcnn,
    'lcnn-lstm': build_lcnn_lstm,
    'resnet': build_resnet,
    # 'resnet-lstm': build_resnet_lstm
}
if model_type not in model_build_map:
    raise ValueError(f'model type "{model_type}" not exist!')

# Replace the path to protcol of ASV2019 depending on your environment.
protocol_tr = "./protocol/train_protocol.csv"
protocol_dev = "./protocol/dev_protocol.csv"
protocol_eval = "./protocol/eval_protocol.csv"

# Choose access type PA or LA.
# Replace 'asvspoof_database/ to your database path.
access_type = "LA"
path_to_database = "/home/tzujohsu/audio-deepfake/" + access_type
path_tr = path_to_database + "/ASVspoof2019_" + access_type + "_train/flac/"
path_dev = path_to_database + "/ASVspoof2019_" + access_type + "_dev/flac/"
path_eval = path_to_database + "/ASVspoof2019_" + access_type + "_eval/flac/"

#%%

if __name__ == "__main__":

    df_tr = pd.read_csv(protocol_tr)
    df_dev = pd.read_csv(protocol_dev)

    
    print("Extracting train data...")
    if os.path.exists(cache + f'/{feature_type}-train.npz'):
        x_train, y_train = load_feature(cache + f'/{feature_type}-train.npz')
    else:
        x_train, y_train = feature_xtract_map[feature_type](df_tr, path_tr, args.datasize)
        if args.savedata: save_feature(x_train, y_train, cache + f'/{feature_type}-train.npz')
    
    x_train, y_train = shuffle(x_train, y_train)
    if args.datasize > 0: x_train, y_train = x_train[:args.datasize], y_train[:args.datasize]
    print(x_train.shape, y_train.shape)
    
    
    print("Extracting dev data...")
    if os.path.exists(cache + f'/{feature_type}-dev.npz'):
        x_val, y_val = load_feature(cache + f'/{feature_type}-dev.npz')
    else:
        x_val, y_val = feature_xtract_map[feature_type](df_dev, path_dev, args.datasize)
        if args.savedata: save_feature(x_val, y_val, cache + f'/{feature_type}-dev.npz')

    x_val, y_val = shuffle(x_val, y_val)
    if args.datasize > 0: x_val, y_val = x_val[:args.datasize], y_val[:args.datasize]
        
    print('Model Building...')
    input_shape = x_train.shape[1:]
    model = model_build_map[model_type](input_shape)

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    es = EarlyStopping(monitor="val_loss", patience=8, verbose=args.verbose)
    cp_cb = ModelCheckpoint(
        filepath=f"{log}/model-{model_type}-{feature_type}-{job_id}.keras",
        monitor="val_loss",
        verbose=args.verbose,
        save_best_only=True,
        mode="auto",
    )
    print("Training start...")
    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=[x_val, y_val],
        callbacks=[es, cp_cb],
        verbose=args.verbose,
    )
    # print("Training history: ", history)

    del x_train, x_val
    print('Training done!')

    # Eval model
    print("Extracting eval data ...")
    df_eval = pd.read_csv(protocol_eval)    
    if args.datasize < 0 and os.path.exists(cache + f'/{feature_type}-eval.npz'):
        x_eval, y_eval = load_feature(cache + f'/{feature_type}-eval.npz')
    else:
        x_eval, y_eval = feature_xtract_map[feature_type](df_eval, path_eval, args.datasize)
        if args.savedata: save_feature(x_eval, y_eval, cache + f'/{feature_type}-eval.npz')
    
    x_eval, y_eval = shuffle(x_eval, y_eval)
    if args.datasize > 0: x_eval, y_eval = x_eval[:args.datasize], y_eval[:args.datasize]

    # predict
    print("Evaluating on eval data ...")
    pred = model.predict(x_eval, verbose = args.verbose)

    score = pred[:, 0] - pred[:, 1]  # Get likelihood
    eer = calculate_eer(y_eval, score)  # Get EER score
    print(f"EER : {eer*100} %")

    y_pred = np.argmax(pred, axis=-1) # 1d array


    accuracy, f1, precision, recall, roc_auc = calculate_classifier_metrics(y_eval, y_pred)
    print(f"accuracy {accuracy}, f1 {f1}, precision {precision}, recall {recall}, roc_auc{roc_auc}")

    results_df = pd.DataFrame({
        'True Label': y_eval,
        'Predicted Label': y_pred,
        'Pred_Prob_0': pred[:, 0],
        'Pred_Prob_1': pred[:, 1]
        })

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'{log}/predictions-{model_type}-{feature_type}-{job_id}.csv', index=False)

    print("You've made it! Have a wonderful day")