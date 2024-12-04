import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from feature import calc_cqt, calc_stft, save_feature
from metrics import calculate_eer, calculate_classifier_metrics

from model.lcnn import build_lcnn
from model.lcnn_lstm import build_lcnn_lstm

import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------------------------------------------------------------------------------------------------------
# cache
cache = '__cache__'
if not os.path.exists(cache):
    os.makedirs(cache)

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
    'stft': calc_stft
}

# model type
model_type = args.model
model_build_map = {
    'lcnn': build_lcnn,
    'lcnn-lstm': build_lcnn_lstm,
}

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
    x_train, y_train = feature_xtract_map[feature_type](df_tr, path_tr, args.datasize)
    
    
    print("Extracting dev data...")
    x_val, y_val = feature_xtract_map[feature_type](df_dev, path_dev, args.datasize)
        
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
        filepath=f"{cache}/model-{model_type}-{feature_type}-{job_id}.keras",
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
    )
    del x_train, x_val
    print('Training done!')

    # Eval model
    print("Extracting eval data ...")
    df_eval = pd.read_csv(protocol_eval)

    
    x_eval, y_eval = feature_xtract_map[feature_type](df_eval, path_eval, args.datasize)

    # predict
    print("Evaluating on eval data ...")
    pred = model.predict(x_eval)

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
    results_df.to_csv(f'{cache}/predictions-{model_type}-{feature_type}-{job_id}.csv', index=False)

    print("You've made it! Have a wonderful day")