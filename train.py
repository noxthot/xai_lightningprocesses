import json
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import time
import torch
import torch.nn as nn

from datetime import datetime
from petastorm.spark import make_spark_converter
from pyspark.sql.functions import lit

import ccc
import utils
import utils_ui


NR_ROWS_USED_TRAIN = None  # Limits the input data to this number of rows. Set to None to use full dataset
NR_ROWS_USED_VAL = None  # Limits the input data to this number of rows. Set to None to use full dataset

MAX_EPOCHS = 1000

SHUFFLING_QUEUE_CAPACITY = 1000

RUNNING_POSTFIX = "_RUNNING"


def train_one_epoch(model, device, train_loader_iter, steps_per_epoch, log_batch_interval, optimizer, epoch, loss_criterion):
    model.train()
    nr_data_rows = 0
    start = time.time()
    running_loss = 0.0

    for batch_idx in range(1, steps_per_epoch + 1):
        batch = next(train_loader_iter)
        data = batch['features'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)

        nr_data_rows += len(data)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        
        if batch_idx % log_batch_interval == 0:
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(epoch, nr_data_rows, loss.item()), flush=True)

    train_loss = running_loss / nr_data_rows

    end = time.time()

    processedRowsPerSec = nr_data_rows / (end - start)

    print('Train Epoch {} [{}] finished with average loss: {:.6f}. Number of processed rows per second: {:.2f}'.format(epoch, nr_data_rows, train_loss, processedRowsPerSec), flush=True)

    return train_loss


def main():
    if ccc.SEED_RANDOM is not None:
        torch.manual_seed(ccc.SEED_RANDOM)
        np.random.seed(ccc.SEED_RANDOM)

    config_model = utils_ui.ask_model_cfg()

    target_mode = config_model["target_mode"]

    model_save_path = os.path.join(ccc.MODEL_ROOT_PATH, f'targetmode_{target_mode}', datetime.now().strftime("%Y_%m_%d__%H-%M"))
    model_running_save_path = model_save_path + RUNNING_POSTFIX

    os.makedirs(model_running_save_path)

    with open(os.path.join(model_running_save_path, "model_cfg.json"), 'w') as f:
        json.dump(config_model, f)   

    norm_fun = utils.getnormfun(config_model["norm_fun"])

    spark = utils.getsparksession()

    config_data, data_path = utils_ui.ask_datacfg(model_running_save_path)

    print(f"Training on data set {data_path}")

    add_val_cols = ["longitude", "latitude", "hour", "day", "month", "year"]
    sel_val_cols = list(set(ccc.TRAIN_COLS + add_val_cols))

    target_col = utils.getTargetCol(target_mode)

    df_train = spark.read.parquet(os.path.join(data_path, 'train'))
    df_val = spark.read.parquet(os.path.join(data_path, 'val'))

    if config_data["datamode"] in {2, 3}:
        df_train = df_train.filter(df_train["year"] != config_data['test_year'])
        df_val = df_val.filter(df_val["year"] != config_data['test_year'])

    df_train = df_train.select(*ccc.TRAIN_COLS + [target_col])
    df_val = df_val.select(*sel_val_cols + [target_col])

    if NR_ROWS_USED_TRAIN is not None:
        df_train = df_train.limit(NR_ROWS_USED_TRAIN)

    if NR_ROWS_USED_VAL is not None:
        df_val = df_val.limit(NR_ROWS_USED_VAL)

    df_train = df_train.withColumn('features', lit(0))  # We need to init column here, otherwise data loader fails
    df_train = df_train.withColumn('label', lit(0))  # We need to init column here, otherwise data loader fails

    df_val = df_val.withColumn('features', lit(0))  # We need to init column here, otherwise data loader fails
    df_val = df_val.withColumn('label', lit(0))  # We need to init column here, otherwise data loader fails

    df_train = df_train.repartition(8)
    df_val = df_val.repartition(2)

    df_train = utils.fit_to_batchsize(df_train)
    df_val = utils.fit_to_batchsize(df_val)

    converter_train = make_spark_converter(df_train)
    converter_val = make_spark_converter(df_val)

    lenconvtrain = len(converter_train)
    lenconvval = len(converter_val)

    print(f"train: {lenconvtrain} samples, val: {lenconvval} samples")

    use_cuda = ccc.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
    devicestr = 'cuda' if use_cuda else 'cpu'
    device = torch.device(devicestr)

    model = utils.Classifier(input_dim=config_model['inputdim'], hidden_layers=config_model['hiddenlayers'], output_dim=config_model['outputdim'], dropout_p=config_model['dropoutp']).to(device, non_blocking=True)
    model.apply(utils.initialize_weights)

    if not use_cuda:
        torch.set_num_threads(ccc.TORCH_NUM_WORKERS)

    optimizer = torch.optim.Adam(model.parameters())

    loss_criterion = nn.BCEWithLogitsLoss(pos_weight=utils.calc_pos_weights(target_mode, device))

    loop_epochs = MAX_EPOCHS
    best_val_loss = float('inf')
    cnt_val_no_improve = 0

    train_monitor = {
                        'train_losses' : [],
                        'val_losses' : [],
                        'val_scores' : [],
                    }

    print(f"Start training using {devicestr}")

    with converter_train.make_torch_dataloader(batch_size=ccc.BATCH_SIZE, transform_spec=utils.get_transform_spec(norm_fun, use_train_cols=ccc.TRAIN_COLS, target_mode=target_mode),
                                                num_epochs=loop_epochs, seed=ccc.SEED_RANDOM, workers_count=ccc.DATALOADER_NUM_WORKERS, 
                                                shuffling_queue_capacity=SHUFFLING_QUEUE_CAPACITY, shuffle_rows=True) as train_dataloader, \
            converter_val.make_torch_dataloader(batch_size=ccc.BATCH_SIZE, transform_spec=utils.get_transform_spec(norm_fun, use_train_cols=ccc.TRAIN_COLS, target_mode=target_mode, return_cols=add_val_cols),
                                                num_epochs=loop_epochs, seed=ccc.SEED_RANDOM,
                                                workers_count=ccc.DATALOADER_NUM_WORKERS) as val_dataloader:                   
        train_dataloader_iter = iter(train_dataloader)
        val_dataloader_iter = iter(val_dataloader)

        shard_count = 1        
        steps_per_epoch_train = len(converter_train) // (ccc.BATCH_SIZE * shard_count)
        steps_per_epoch_val = len(converter_val) // (ccc.BATCH_SIZE * shard_count)

        epochs = range(1, loop_epochs + 1)

        for epoch in epochs:
            print(f"Start epoch {epoch}.", flush=True)
            train_loss = train_one_epoch(model, device, train_dataloader_iter, steps_per_epoch_train, ccc.LOG_BATCH_INTERVAL, optimizer, epoch, loss_criterion)
            
            val_loss, _, val_score = utils.test_one_epoch(model, device, val_dataloader_iter, steps_per_epoch_val, loss_criterion, target_mode=target_mode)
            val_score["epoch"] = epoch

            train_monitor['train_losses'].append(train_loss)
            train_monitor['val_losses'].append(val_loss)
            train_monitor['val_scores'].append(val_score)

            with open(os.path.join(model_running_save_path, "train_monitor.pickle"), 'wb') as handle:
                pickle.dump(train_monitor, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(model_running_save_path, "train_monitor.json"), 'w') as handle:
                json.dump(train_monitor, handle, indent=4)

            myplotdata = pd.DataFrame(list(zip(epochs, train_monitor["train_losses"], train_monitor["val_losses"])), columns = ["epochs", "train_losses", "val_losses"])
            myplotdata = myplotdata.melt(var_name="cols", value_name="vals", id_vars="epochs")

            myplot = sns.lineplot(data=myplotdata, x="epochs", y="vals", hue="cols")

            myplot.get_figure().savefig(os.path.join(model_running_save_path, "lossplot.png"))
            myplot.get_figure().clf()

            # Early Stopping
            if val_loss < best_val_loss:
                cnt_val_no_improve = 0
                best_val_loss = val_loss
                bad_model_prefix = ""
                print(f"NEW BEST MODEL - VAL: {val_loss}")
            else:
                cnt_val_no_improve += 1
                bad_model_prefix = "_"


            epochstr = str(epoch).zfill(5)

            torch.save({
                    'input_dim': config_model["inputdim"],
                    'hidden_layers': config_model["hiddenlayers"],
                    'output_dim': config_model["outputdim"],
                    'dropout_p': config_model["dropoutp"],
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, 
                os.path.join(model_running_save_path, f"{bad_model_prefix}model_{epochstr}.pt")
            )

            esp = config_model["earlystoppingpatience"]

            if cnt_val_no_improve >= esp:
                print(f"Model did not improve for {esp} epochs -> STOPPING")
                break

    converter_train.delete()
    converter_val.delete()

    os.rename(model_running_save_path, model_save_path)

    print("FIN.")


if __name__ == '__main__':
    main()
