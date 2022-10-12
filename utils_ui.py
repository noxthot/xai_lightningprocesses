import ccc
import git
import inquirer
import json
import os


def ask_targetmode():
    questions = [inquirer.List('targetmode',
                            message="Target Mode?",
                            choices=[
                                        "1: Flash",
                                        "2: Flash windowed sum",
                                        "3: Flash windowed max",
                                    ]
                            )
                ]

    answers = inquirer.prompt(questions)

    return int(answers["targetmode"][0])


def ask_modeldir(target_mode):
    tmsubdir = f'targetmode_{target_mode}'
    model_root_tm_path = os.path.join(ccc.MODEL_ROOT_PATH, tmsubdir)

    modeldirs = os.listdir(model_root_tm_path)
    modeldirs.sort(reverse=True)

    questions = [inquirer.List('modeldir',
                            message="Which model should be tested?",
                            choices=modeldirs)
    ]

    answers = inquirer.prompt(questions)

    return [os.path.join(tmsubdir, answers["modeldir"]), os.path.join(model_root_tm_path, answers["modeldir"])]


def ask_model_cfg():
    questions = [
        inquirer.Text(
            'hiddenlayers',
            message="Give a list of nodes per hiddenlayers separated by comma. E.g.: 512,512,3",
        ),
        inquirer.Text(
            'dropoutp',
            message="Choose a dropout value. E.g. 0.1",
        ),
        inquirer.Text(
            'earlystoppingpatience',
            message="Choose early stopping patience. E.g. 5",
        ),
        inquirer.List(
            'normfun',
            message="Choose a normalization method",
            choices=["minmax", "meanstd", "disabled"],
        ),
    ]
    answers = inquirer.prompt(questions)

    dropoutp = float(answers["dropoutp"])
    norm_fun = answers["normfun"]
    esp = int(answers["earlystoppingpatience"])
    hiddenlayers = [int(val) for val in answers["hiddenlayers"].split(',')]

    target_mode = ask_targetmode()

    if target_mode in {1, 3}:
        outputdim = 1
    elif target_mode == 2:
        outputdim = 3
    else:
        raise Exception(f"Target mode {target_mode} unknown")

    config_model = dict()
    config_model["dropoutp"] = dropoutp
    config_model["earlystoppingpatience"] = esp
    config_model["hiddenlayers"] = hiddenlayers
    config_model["inputdim"] = 671
    config_model["outputdim"] = outputdim
    config_model["target_mode"] = target_mode
    config_model["norm_fun"] = norm_fun

    return config_model


'''
User UI for building data config. In case store_settings_path is given, git diff and data_cfg is stored in that path
'''
def ask_datacfg(store_settings_path=""):
    data_cfg = dict()
    data_cfg['datamode'] = 3  # hardcoded right now, because i don't see why we should use others

    dmsubdir = f"datamode_{data_cfg['datamode']}"
    data_root_dm_path = os.path.join(ccc.DATA_ROOT_PATH, dmsubdir)

    datadirs = os.listdir(data_root_dm_path)
    datadirs.sort(reverse=True)

    question_datadir = [
        inquirer.List(
            'datasource',
            message="Choose a data source",
            choices=datadirs,
        ),
    ]

    data_cfg['datasource'] = inquirer.prompt(question_datadir)['datasource']

    if data_cfg['datamode'] in {2, 3}:
        train_data_path = os.path.join(data_root_dm_path, data_cfg['datasource'], 'train')
        yeardirs = os.listdir(train_data_path)

        yeardirs = [dir.split("=")[1] for dir in yeardirs if dir.startswith("year=")]
        yeardirs.sort(reverse=True)

        question_testdir = [
            inquirer.List(
                'test_year',
                message="Choose a test year",
                choices=yeardirs,
            ),
        ]

        data_cfg['test_year'] = inquirer.prompt(question_testdir)['test_year']

    # get and store git hash    
    repo = git.Repo(search_parent_directories=True)
    data_cfg['git_hash'] = repo.head.object.hexsha

    diffs = repo.git.diff('HEAD')

    if diffs != "":
        data_cfg['git_hash'] += "_dirty"

        if store_settings_path != "":
            with open(os.path.join(store_settings_path, 'gitdiff.txt'), 'w') as f:
                f.write(diffs)

    if store_settings_path != "":
        with open(os.path.join(store_settings_path, "data_cfg.json"), 'w') as f:
            json.dump(data_cfg, f)

    return [data_cfg, os.path.join(data_root_dm_path, data_cfg['datasource'])]