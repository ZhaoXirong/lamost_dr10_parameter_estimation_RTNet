from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# np.object = object
# np.bool = bool
import os
import torch.utils.data as Data
import argparse


from tqdm import tqdm
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.tensorboard import SummaryWriter

from models.RRNet import RRNet


parser = argparse.ArgumentParser()

parser.add_argument(
    '--train', type=bool, default=True,
    help='Whether to enable the training function for the program.',
)

parser.add_argument(
    '--predict', type=bool, default=True,
    help='Whether to enable the prediction function for the program.',
)

parser.add_argument(
    '--net', choices=['RRNet'], default='RRNet',
    help='The models you need to use.',
)
parser.add_argument(
    '--mode', choices=['raw', 'pre-RNN', 'post-RNN'], default='post-RNN',
    help='The mode of the RRN embedding.',
)
parser.add_argument(
    '--list_ResBlock_inplanes', type=list, default=[4, 8, 16],  # 原[4,8,16]
    help='The size of inplane in the RRNet residual block.'
)
parser.add_argument(
    '--n_rnn_sequence', type=int, default=30,   # 40
    help='The number of RRN sequences.'
)
parser.add_argument(
    '--path_reference_set', type=str, default='./data/DR8_Preprocessing',
    help='The path of the reference set.',
)
parser.add_argument(
    '--path_labels', type=str, default='./data/LABELS',
    help='The path of the labels.',
)
parser.add_argument(
    '--path_log', type=str, default='./model_log/',
    help='The path to save the model data after training.'
)

parser.add_argument(
    '--path_preprocessed', type=str, default='./data/data_Preprocessed',   # D:\文\jupyter\My\data\data_Preprocessed  .data/data_Preprocessed
    help='The path to save the model data after training.'
)

parser.add_argument(
    '--batch_size', type=int, default=256,
    help ='The size of the batch.'
)
parser.add_argument(
    '--n_epochs', type=int, default=30,
    help='Number of epochs to train.'
)
parser.add_argument(
    '--noise_model', type=bool, default=True,    # add_training_noise
    help='Train: Whether to add Gaussian noise with a mean of 1 and a variance of 1 during training    Predict: Whether to use a model trained with noise '
)
parser.add_argument(
    '--DeepEnsemble', type=bool, default=True,
    help='Whether to use a fine-tuning model'
)
parser.add_argument(
    # 实际标签：['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
    #                                      'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg']
    '--label_list', type=list, default=['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
                                          'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg'],    # 'Teff[K]', 'Logg', 'FeH','CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH','KH', 'CaH', 'TiH', 'CrH', 'MnH', 'FeH', 'NiH', 'CuH'
    help='The label data that needs to be learned.'
)



def get_dataset_info(args):
    dir_list = os.listdir(args.path_preprocessed)
    print(dir_list)
    if dir_list == ['label_config.pkl', 'test_flux.pkl', 'test_label.csv', 'train_flux.pkl', 'train_label.csv', 'valid_flux.pkl', 'valid_label.csv']:
        # 加载数据
        flux = np.load(args.path_reference_set + "/BR_Flux_Preprocessing_lamost_apogee_between_5_50.npy")
        label = np.load(args.path_labels + "/between_5_50.npy", allow_pickle=True)
        label = pd.DataFrame(label, columns=['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
                                         'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg'])
        # 将原始温度标签（单位为K）转换为以10为底的对数值。原始标签的数值范围从 3000-60000 K 缩小到了 3.5-4.8 之间
        label['Teff[K]'] = np.log10(label['Teff[K]'].astype(np.float64))

        # print("label['Teff[K]']:", label['Teff[K]'])
        # print("label", label)

        # 标签标准化
        std_label = np.sqrt(label.iloc[:, :17].var())
        mean_label = label.iloc[:, :17].mean()
        # label_std = (label.iloc[:, :17] - label.iloc[:, :17].mean()) / np.sqrt(label.iloc[:, :17].var())
        # label_std['snrg'] = label['snrg']
        # print("label", label_std)
        # print("label,std\n", std_label, "\nlabel,std\n", mean_label)

        # 划分数据集

        x_train, x_valid_test, y_train, y_valid_test = train_test_split(flux, label, test_size=0.3, random_state=123)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.67, random_state=123)

        print('size:', x_train.shape, y_train.shape,)
        print('Training set:', x_train.shape[0], 'samples', 'flux type:', type(x_train), 'Training labels:', type(y_train))
        print('Validation set:', x_valid.shape[0], 'samples')
        print('Testing set:', x_test.shape[0], 'samples')

        # 数据集标准化
        # 训练集
        # Training set normalization
        flux_3sigma_sc = StandardScaler()
        x_train_T = flux_3sigma_sc.fit_transform(x_train.T)   # 对每条光谱数据进行标准化
        x_train = x_train_T.T

        # 验证集
        flux_3sigma_sc3 = StandardScaler()
        x_valid_T = flux_3sigma_sc3.fit_transform(x_valid.T)
        x_valid = x_valid_T.T

        # 测试集
        flux_3sigma_sc3 = StandardScaler()
        x_test_T = flux_3sigma_sc3.fit_transform(x_test.T)
        x_test = x_test_T.T

        # 填充空值
        # Fill null
        pd.DataFrame(x_train).fillna(1, inplace=True)
        # 检查空值
        # Check Null
        # print(pd.isnull(x_train).any())
        # print(pd.isnull(y_train).any())
        # print(pd.isnull(x_valid).any())
        # print(pd.isnull(y_valid).any())
        # print(pd.isnull(x_test).any())
        # print(pd.isnull(y_test).any())

        # X_train_torch = (pickle.load(open(args.path_reference_set + "train_flux.pkl", 'rb')) - flux_mean) / flux_std
        # X_valid_torch = (pickle.load(open(args.path_reference_set + "valid_flux.pkl", 'rb')) - flux_mean) / flux_std

        # 保存
        label_config = {
            "label_list": ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
                                              'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg'],
            "label_mean": mean_label,
            "label_std": std_label,
        }

        with open(os.path.join(args.path_preprocessed, 'label_config.pkl'), 'wb') as f:
            pickle.dump(label_config, f)
        with open(os.path.join(args.path_preprocessed, 'train_flux.pkl'), 'wb') as f:
            pickle.dump(x_train, f)
        with open(os.path.join(args.path_preprocessed, 'valid_flux.pkl'), 'wb') as f:
            pickle.dump(x_valid, f)
        with open(os.path.join(args.path_preprocessed, 'test_flux.pkl'), 'wb') as f:
            pickle.dump(x_test, f)
        y_train.to_csv(os.path.join(args.path_preprocessed, 'train_label.csv'), index=False)
        y_valid.to_csv(os.path.join(args.path_preprocessed, 'valid_label.csv'), index=False)
        y_test.to_csv(os.path.join(args.path_preprocessed, 'test_label.csv'), index=False)
    else:
        label_config = pickle.load(open(args.path_preprocessed + "/label_config.pkl", 'rb'))
        mean_label = label_config["label_mean"]
        std_label = label_config["label_std"]

        x_train = pickle.load(open(args.path_preprocessed + "/train_flux.pkl", 'rb'))
        x_valid = pickle.load(open(args.path_preprocessed + "/valid_flux.pkl", 'rb'))
        x_test = pickle.load(open(args.path_preprocessed + "/test_flux.pkl", 'rb'))
        y_train = pd.read_csv(os.path.join(args.path_preprocessed, 'train_label.csv'))
        y_valid = pd.read_csv(os.path.join(args.path_preprocessed, 'valid_label.csv'))
        y_test = pd.read_csv(os.path.join(args.path_preprocessed, 'test_label.csv'))
        del label_config

    X_train_torch = torch.tensor(x_train, dtype=torch.float32)
    X_valid_torch = torch.tensor(x_valid, dtype=torch.float32)
    X_test_torch = torch.tensor(x_test, dtype=torch.float32)

    y_train_torch = y_train[args.label_list].values
    y_valid_torch = y_valid[args.label_list].values
    y_test_torch = y_test[args.label_list].values

    y_train_torch = torch.tensor(y_train_torch, dtype=torch.float32)
    y_valid_torch = torch.tensor(y_valid_torch, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test_torch, dtype=torch.float32)

    print("x_train_torch.shape:", X_train_torch.shape)
    print("x_train_torch.dtype:", X_train_torch.dtype)
    print("y_train_torch.shape:", y_train_torch.shape)
    print("y_train_torch.dtype:", y_train_torch.dtype)

    print("x_train_torch:\n", X_train_torch)
    print("y_train_torch:\n", y_train_torch)

    train_dataset = Data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = Data.TensorDataset(X_valid_torch, y_valid_torch)
    test_dataset = Data.TensorDataset(X_test_torch, y_test_torch)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # True
        num_workers=0,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    mean_label = mean_label[args.label_list].values
    std_label = std_label[args.label_list].values
    print("label,std\n", std_label, "\nlabel,std\n", mean_label)
    dataset_info = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "label_mean": mean_label,
        "label_std": std_label,
        "test_loader": test_loader,
        "test_labels": y_test
    }

    return dataset_info

def add_noise(input_x):
    normal_noise = torch.normal(mean=torch.zeros_like(input_x), std=torch.ones_like(input_x))
    normal_prob = torch.rand_like(input_x)
    input_x[normal_prob<0.25] += normal_noise[normal_prob<0.25]     # 随机扰动  normal_prob<0.25相当于生成一个布尔型张量

    return input_x

def train(args, dataset_info, train_label=['Teff[K]', 'Logg', 'FeH'], model_number="SP0", cuda=True):
    if args.net == "RRNet":
        if args.mode != "raw":
            # eg: RRNet(Nr=[16-32-64]-Ns=3)_post-RNN
            model_name = "RRNet(Nr=[%s]-Ns=%d)_%s" % (
            '-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.n_rnn_sequence, args.mode)

        else:
            # eg: RNet(Nr=[16-32-64])_raw
            model_name = "RRNet(Nr=[%s])_%s" % ('-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.mode)

        # print("model name : ", model_name)
        net = RRNet(
            mode=args.mode,
            num_lable=len(train_label),
            list_ResBlock_inplanes=args.list_ResBlock_inplanes,
            num_rnn_sequence=args.n_rnn_sequence,
            len_spectrum=3450,
        )
        if cuda:
            net = net.to("cuda")
    else:
        raise Exception("模型名字错误，程序已终止。")

    if args.noise_model:
        model_name += "_add-noise"

    model_name += '/' + model_number

    # print(net)
    print(model_name)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)  # 优化器选择Adam

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 用学习率调整器动态调整优化器的学习率

    log_dir = args.path_log + model_name  # 日志
    writer = SummaryWriter(log_dir=log_dir)

    label_index = [args.label_list.index(i) for i in train_label]   # 把标签转成数值

    print("label_index: ", label_index)

    best_loss = np.inf
    # Iterative optimization

    #保存验证集

    output_label = np.zeros(shape=(len(dataset_info["valid_loader"].dataset), len(train_label)))
    output_label_err = np.zeros_like(output_label)

    for epoch in tqdm(range(1, args.n_epochs + 1)):
        net.train()
        torch.cuda.empty_cache()

        train_mae = np.zeros(len(train_label))
        train_loss = 0.0    # 记录模型训练过程中的累计损失（所有批次数据的损失之和）
        # Train
        for step, (batch_x, batch_y) in enumerate(dataset_info["train_loader"]):

            if args.noise_model and epoch > 5:
                batch_x = add_noise(batch_x)

            batch_y = batch_y[:, label_index]
            if cuda:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
            mu, sigma = net(batch_x)
            loss = net.get_loss(batch_y, mu, sigma)

            train_loss += loss.to("cpu").data.numpy()


            optimizer.zero_grad()       # 清空梯度
            loss.backward()             # 反向传播，计算梯度
            optimizer.step()            # 更新梯度

            n_iter = (epoch - 1) * len(dataset_info["train_loader"]) + step + 1  # 局迭代步数n_iter，用于记录 Tensorboard 日志及其他训练过程中的信息。

            mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][
                label_index]                              # 反归一化处理 得到真实的预测值
            batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + \
                      dataset_info["label_mean"][label_index]   # 反归一化处理 得到真实的标签

            mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')   # 计算平均绝对误差（MAE）
            train_mae += mae

            writer.add_scalar('Train/loss', loss.to("cpu").data.numpy(), n_iter)
            for i in range(len(train_label)):
                writer.add_scalar('Train/%s_MAE' % train_label[i], mae[i], n_iter)

        scheduler.step()    # 更新优化器的学习率
        lr = optimizer.state_dict()['param_groups'][0]['lr']   # 返回优化器的状态字典

        train_loss /= (step + 1)   # train_loss/N 平均训练损失值
        train_mae /= (step + 1)   # 平均 MAE

        torch.cuda.empty_cache()
        net.eval()         # 将网络设置为评估模式

        valid_mae = np.zeros(len(label_index))
        vlaid_diff_std = np.zeros(len(label_index))
        valid_loss = 0.0



        #保存
        save = False
        if epoch == args.n_epochs:
            save = True
        # Valid
        for step, (batch_x, batch_y) in enumerate(dataset_info["valid_loader"]):

            with torch.no_grad():  # 上下文管理器:代码块中所有的计算都不会计算梯度
                batch_y = batch_y[:, label_index]
                if cuda:
                    batch_x = batch_x.to("cuda")
                    batch_y = batch_y.to("cuda")
                mu, sigma = net(batch_x)

                if save:
                    output_label[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = mu.to("cpu").data.numpy()
                    output_label_err[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = batch_y.to("cpu").data.numpy()

                loss = net.get_loss(batch_y, mu, sigma)

                valid_loss += loss.to("cpu").data.numpy()

                n_iter = (epoch - 1) * len(dataset_info["valid_loader"]) + step + 1

                sigma = np.sqrt(sigma.to("cpu").data.numpy()) * dataset_info["label_std"][label_index]

                mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][
                    label_index]
                batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + \
                          dataset_info["label_mean"][label_index]

                # if save:
                #     output_label[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = mu
                #     output_label_err[step * dataset_info["valid_loader"].batch_size:step * dataset_info["valid_loader"].batch_size + mu.shape[0]] = batch_y

                diff_std = (mu - batch_y).std(axis=0)  # 计算每列数据的标准差
                sigma_mean = sigma.mean(axis=0)        # 计算每列数据的平均值

                vlaid_diff_std += diff_std

                mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')    # 用sklearn计算mae

                valid_mae += mae

            writer.add_scalar('Valid/loss', loss.to("cpu").data.numpy(), n_iter)
            for i in range(len(train_label)):
                writer.add_scalar('Valid/%s_MAE' % train_label[i], mae[i], n_iter)
                writer.add_scalar('Valid/%s_diff_std' % train_label[i], diff_std[i], n_iter)
                writer.add_scalar('Valid/%s_sigma' % train_label[i], sigma_mean[i], n_iter)

        valid_loss /= (step + 1)
        valid_mae /= (step + 1)
        vlaid_diff_std /= (step + 1)

        torch.save(net.state_dict(), log_dir + '/weight_temp.pkl')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), log_dir + '/weight_best.pkl')

        print("EPOCH %d | lr %f | train_loss %.4f | valid_loss %.4f" % (epoch, lr, train_loss, valid_loss),
              "| valid_mae", valid_mae,
              "| valid_diff_std", vlaid_diff_std)

    out_mu = np.array(output_label)
    out_sigma = np.array(output_label_err)
    df = pd.read_csv(args.path_preprocessed + '/valid_label.csv')
    for i in range(len(train_label)):
        df["%s_%s" % (model_name, train_label[i])] = out_mu[:, i]
        df["%s_%s_err" % (model_name, train_label[i])] = out_sigma[:, i]
    df.to_csv(args.path_preprocessed + '/valid_label_mode_out.csv', index=False)


def predict(args, dataset_info):
    def one_predict(args, test_loader, model_path):
        print(model_path)

        train_label = ['Teff[K]', 'Logg', 'FeH'] if model_path.split("/")[-1][:2] == "SP" else ['CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH','KH', 'CaH', 'TiH', 'CrH','MnH', 'NiH']

        if args.net == "RRNet":
            net = RRNet(
                mode=args.mode,
                num_lable=len(train_label),
                list_ResBlock_inplanes=args.list_ResBlock_inplanes,
                num_rnn_sequence=args.n_rnn_sequence,
                len_spectrum=3450,
            ).to("cuda")
        else:
            pass
        net.eval()
        net.load_state_dict(torch.load(model_path + "/weight_best.pkl"))

        output_label = np.zeros(shape=(len(test_loader.dataset), len(train_label)))
        output_label_err = np.zeros_like(output_label)
        for step, batch in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                batch_x = batch[0]

                mu, sigma = net(batch_x.to("cuda"))
                mu = mu.to("cpu").data.numpy()
                sigma = np.sqrt(sigma.to("cpu").data.numpy())

            output_label[step * test_loader.batch_size:step * test_loader.batch_size + mu.shape[0]] = mu
            output_label_err[step * test_loader.batch_size:step * test_loader.batch_size + mu.shape[0]] = sigma

        return [output_label, output_label_err]

    label_mean = dataset_info["label_mean"]
    label_std = dataset_info["label_std"]
    test_loader = dataset_info["test_loader"]

    if args.net == "RRNet":
        if args.mode != "raw":
            model_name = "RRNet(Nr=[%s]-Ns=%d)_%s" % (
            '-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.n_rnn_sequence, args.mode)
        else:
            model_name = "RRNet(Nr=[%s])_%s" % ('-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.mode)
    elif args.net == "StarNet":
        model_name = "StarNet_%s" % args.mode
    if args.noise_model:
        model_name += "_add-noise"

    model_path = args.path_log + model_name
    model_list = os.listdir(model_path)
    output_list_SP = []
    output_list_CA = []
    if not args.DeepEnsemble:
        if "SP0" in model_list:
            output_list_SP.append(one_predict(args, test_loader, model_path=model_path + "/SP0"))
        if "CA0" in model_list:
            output_list_CA.append(one_predict(args, test_loader, model_path=model_path + "/CA0"))
    else:
        for model in model_list:
            out = one_predict(args, test_loader, model_path=model_path + "/" + model)
            if model[:2] == "SP":
                output_list_SP.append(out)
            elif model[:2] == "CA":
                output_list_CA.append(out)

    mu_list = []     # 经过模型输出的mu
    sigma_list = []   # 经过模型输出的sigma
    for i in range(min(len(output_list_SP), len(output_list_CA))):
        mu_list.append(np.hstack((output_list_SP[i][0], output_list_CA[i][0])))  # np.hstack: 将两个数组沿着水平方向进行拼接
        sigma_list.append(np.hstack((output_list_SP[i][1], output_list_CA[i][1])))

    del output_list_SP, output_list_CA
    mu_list = np.array(mu_list)
    sigma_list = np.array(sigma_list)

    out_mu = mu_list.mean(0)
    out_sigma = ((mu_list ** 2 + sigma_list ** 2)).mean(0) - out_mu ** 2   # 计算一组正态分布的平均方差
    out_sigma = np.sqrt(out_sigma)               # 平均标准差

    train_label = ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
                                          'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH']
    train_label_index = [args.label_list.index(i) for i in train_label]

    out_mu = out_mu * label_std[train_label_index] + label_mean[train_label_index]
    out_sigma *= label_std[train_label_index]

    del mu_list, sigma_list

    if args.path_log is not None:
        print(dataset_info["test_labels"])
        true_mu = pd.DataFrame(dataset_info["test_labels"], columns=train_label)
        true_mu = true_mu* label_std[train_label_index] + label_mean[train_label_index]
        diff_std = (true_mu - out_mu).std(axis=0)
        mae = mean_absolute_error(true_mu, out_mu, multioutput='raw_values')
        print(
            "mae:", mae,
            "diff_std", diff_std,
        )
        df = pd.read_csv(args.path_preprocessed+'/test_label.csv')
        cols = [col for col in df.columns if col in train_label]
        df[cols] = df[cols] * label_std[train_label_index] + label_mean[train_label_index]
        for i in range(len(train_label)):
            df["%s_%s" % (model_name, train_label[i])] = out_mu[:, i]
            df["%s_%s_err" % (model_name, train_label[i])] = out_sigma[:, i]
        df.to_csv(args.path_preprocessed+'/test_label.csv'[:-4] + "_%s_out.csv" % model_name, index=False)



if __name__ == "__main__":

    args = parser.parse_args()
    dataset_info = get_dataset_info(args)

    if args.train:
        train(args, dataset_info=dataset_info,
              train_label=['Teff[K]', 'Logg', 'FeH'],
              model_number="SP2",
              cuda=True)
        # ['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
         #                                      'KH', 'CaH', 'TiH', 'CrH','MnH', 'FeH', 'NiH', 'snrg']
        train(args, dataset_info=dataset_info,
              train_label=['CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH'],
              model_number="CA2",
              cuda=True)

    if args.predict:
        predict(args, dataset_info)
