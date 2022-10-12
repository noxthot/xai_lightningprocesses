import ccc
import json
import os
import pickle
import torch
import utils
import utils_ui

import torch.nn as nn

from petastorm.spark import make_spark_converter

NR_CHUNKS = 19  # This has to be a divider of lenconvtest; otherwise we lose samples or look at some samples twice
NR_ROWS_USED_TEST = None  # Limits the input data to this number of rows. Set to None to use full dataset
USE_EPOCH = None

INCLUDE_HIDDEN_LAYERS = False
HOOK_ACT_NAME = 'hlayer_'

def main():
    target_mode = utils_ui.ask_targetmode()
    model_dir, model_path = utils_ui.ask_modeldir(target_mode)

    with open(os.path.join(model_path, 'data_cfg.json'), 'r') as f:
        config_data = json.load(f)

    with open(os.path.join(model_path, 'model_cfg.json'), 'r') as f:
        config_model = json.load(f)

    with open(os.path.join(model_path, 'train_monitor.pickle'), 'rb') as f:
        train_monitor = pickle.load(f)

    opt_threshold, best_epoch = utils.getOptThresholdFromVal(train_monitor, USE_EPOCH)

    target_col = utils.getTargetCol(config_model["target_mode"])

    add_cols = ["longitude", "latitude", "hour", "day", "month", "year"]
    sel_cols = list(set(ccc.TRAIN_COLS + add_cols + [target_col]))

    df_test = utils.get_testdf_spark(config_data, sel_cols, NR_ROWS_USED_TEST)

    converter_test = make_spark_converter(df_test)
    lenconvtest = len(converter_test)

    if lenconvtest % NR_CHUNKS != 0:
        raise Exception(f"NR_ITERATIONS (= {NR_CHUNKS}) has to be a divider of lenconvtest (= {lenconvtest}).")

    print(f"Test dataset: {lenconvtest} samples")

    norm_fun = utils.getnormfun(config_model["norm_fun"])

    use_cuda = ccc.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
    devicestr = 'cuda' if use_cuda else 'cpu'
    device = torch.device(devicestr)

    if not use_cuda:
        torch.set_num_threads(ccc.TORCH_NUM_WORKERS)
        
    model, model_name = utils.load_model(model_dir, device, best_epoch)
    hook_name_list = []

    if INCLUDE_HIDDEN_LAYERS:
        for idx in range(len(model.hidden_layers)):
            hook_name_list.append(f"{HOOK_ACT_NAME}_{idx}")
            model.hidden_layers[idx].register_forward_hook(utils.get_activation(f"{HOOK_ACT_NAME}_{idx}"))

    loss_criterion = nn.BCEWithLogitsLoss(pos_weight=utils.calc_pos_weights(config_model["target_mode"], device))

    with converter_test.make_torch_dataloader(batch_size=(lenconvtest / NR_CHUNKS), transform_spec=utils.get_transform_spec(norm_fun, use_train_cols=ccc.TRAIN_COLS, return_cols=add_cols, target_mode=config_model["target_mode"]),
                                              num_epochs=1, workers_count=ccc.DATALOADER_NUM_WORKERS) as test_dataloader:
        test_dataloader_iter = iter(test_dataloader)

        test_loss, test_df, test_score = utils.test_one_epoch(model, device, test_dataloader_iter, NR_CHUNKS, loss_criterion, config_model["target_mode"], hook_name_list, threshold=opt_threshold)

    print(f"Final loss: {test_loss}")

    with open(os.path.join(model_path, f"{model_name}_test_df.pickle"), 'wb') as handle:
        pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    json_test_score = json.dumps(test_score, indent=4)
    json_file = open(os.path.join(model_path, f"{model_name}_test_scores.json"), 'w')
    json_file.write(json_test_score)
    json_file.close()

    print("FIN.")

if __name__ == '__main__':
    main()
