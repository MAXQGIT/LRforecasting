import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import torch.nn as nn
from torch.nn import init
from chinese_calendar import is_holiday  # 中国节假日数据库

warnings.simplefilter('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图中文设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Matplotlib(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df_load = pd.read_excel(self.data_path, sheet_name=0, header=None)
        self.df_weather = pd.read_excel(self.data_path, sheet_name=1, header=None)

    # 绘制各类气象条件分类汇总后的柱形图
    def weather_type_pitcture(self):
        plt.figure()
        weather_type = self.df_weather[1].value_counts(sort=True)  # 将数据中各类天气分类汇总
        weather_type.plot.bar()  # 绘制柱状图
        plt.savefig('../img/weather_type.png')  # 图片保存路径
        plt.show()  # 图片展示

    # 绘制各类气象条件分类汇总后的柱形图
    def weather_type2_picture(self):
        # 删除一些不需要展示的种类
        df_weather = self.df_weather.drop(self.df_weather[self.df_weather[1] == "天气类型"].index)
        df_weather = df_weather.drop(df_weather[df_weather[1] == "风向"].index)
        df_weather = df_weather.drop(df_weather[df_weather[1] == "风速"].index)
        df_weather = df_weather.drop(df_weather[df_weather[1] == "降雨量"].index)
        weather_type = df_weather[1].value_counts(sort=True)  # 将数据中各类天气分类汇总
        weather_type.plot.bar()  # 绘制柱状图
        plt.savefig('../img/weather_type2.png')  # 图片保存路径
        plt.show()  # 图片展示

    # 绘制箱线图
    def boxplot_picture(self, df_load):
        column_list = list(range(1, 97))
        plt.figure(figsize=(18, 6))
        df_load.boxplot(column=column_list, figsize=(20, 4))
        plt.savefig('../img/load_boxplot.png')
        plt.show()

    def raw_data_plot(self):
        self.weather_type_pitcture()
        self.weather_type2_picture()

    # 绘制模型训练参数趋势
    def train_model_plot(self, train_ls, test_ls, train_accus, test_accus):
        plt.figure(figsize=(10, 8), dpi=100)
        plt.plot(train_ls, label='train')
        plt.plot(test_ls, label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.title('Train and test mse curve')
        plt.savefig('../img/train_test_mse.png')
        plt.show()

        plt.figure(figsize=(10, 8), dpi=100)
        plt.plot(train_accus, label='train')
        plt.plot(test_accus, label='test')
        plt.xlabel('epoch')
        plt.ylabel('accuary')
        plt.title('Train and test accuary curve')
        plt.legend()
        plt.savefig('../img/train_test_accu.png')
        plt.show()

    # 验证模型图形
    def eval_plot(self, precision):
        something_wrong = np.where(precision < 0.95)[0]
        plt.figure(figsize=(10, 8))
        plt.boxplot(precision)
        plt.title('Test accuary')
        plt.savefig('../img/accu_box.png')
        plt.show()
        print(np.array(precision).mean())
        print((something_wrong))
        plt.figure(figsize=(12, 8))
        plt.scatter(range(155), precision)
        plt.title('Test accuary')
        plt.savefig('../img/accu_scatter.png')
        plt.show()

    def predicted_real_values_plot(self, y_pred, y_real, precision):
        y_pred = y_pred.reshape((-1, 1))
        y_real = y_real.reshape((-1, 1))
        plt.figure(figsize=(200, 8))
        plt.plot(y_real * 7000, label='real')
        plt.plot(y_pred * 7000, label='pred')
        plt.xticks(np.arange(0, 96 * 154, 96))
        plt.legend()
        plt.title('Comparison of predicted and real values')
        plt.savefig('../img/real_pred.png')
        plt.show()

        print_index = precision.argmax()
        print(y_real[print_index].reshape((-1, 1)) * 7000)
        print(y_pred[print_index].reshape((-1, 1)) * 7000)
        plt.figure()
        plt.plot(y_real[:print_index].reshape((-1, 1)) * 7000, label='real')
        plt.plot(y_pred[:print_index].reshape((-1, 1)) * 7000, label='pred')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('load')
        plt.grid()
        plt.savefig('../img/real_pred1.png')
        print(precision[print_index])
        plt.show()


class MODEL(Matplotlib):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 空值填充
    def nan_value_fill(self, df_load, df_weather2):
        for i in range(1, 5):
            df_load[i][df_load[i] == 0] = df_load[i].median()  # 将每列中等于0的数据转换成该列的中位数

        # print(df_load.isna().sum())
        df_load = df_load.fillna(axis=0, method='ffill')  # 按列填充空值。 ffill是将前一个非空缺值进行填充。 ffill是forward fill的缩写。
        df_data = pd.merge(df_load, df_weather2, how='left', on=[0, 0])  # 两个表使用左连接的方式进行合并
        for column in ['max_tempe', 'min_tempe', 'avg_tempe', 'humidity']:
            index = df_data[column][df_data[column].isna()].index
            for idx in index:
                if idx - 365 > 0:
                    df_data.at[idx, column] = df_data.at[idx - 365, column]
                else:
                    df_data.at[idx, column] = df_data.at[idx + 2 * 365, column]
        return df_data

    # 简单数据数据分析方法
    def data_analy(self, df_weather2):
        a = df_weather2.isna().sum()  # 将数据中不带有空行的数据进项求和
        print('各类天气数据求和结果:', a)
        b = df_weather2.describe()  # 数据中众数，平均数，求和，方差等一系列的数据分析结果
        print('数据分析结果:', b)

    # 数据处理
    def data_clear(self):
        # 对原始数据进行数据分析绘图
        # self.raw_data_plot()
        # 将各类数据按照指定类型抽取  选取指定类型天气的时间和对应的数据
        df_max_tempe = self.df_weather.loc[self.df_weather[1] == "最高温度", [0, 2]]
        df_min_tempe = self.df_weather.loc[self.df_weather[1] == "最低温度", [0, 2]]
        df_avg_tempe = self.df_weather.loc[self.df_weather[1] == "平均温度", [0, 2]]
        df_humidity = self.df_weather.loc[self.df_weather[1] == "湿度", [0, 2]]
        # 将抽取出去的各类型天气对应的数据根据时间一一对应
        df_weather2 = pd.merge(df_max_tempe, df_min_tempe, how='left', on=[0, 0])
        df_weather2 = pd.merge(df_weather2, df_avg_tempe, how='left', on=[0, 0])
        df_weather2 = pd.merge(df_weather2, df_humidity, how='left', on=[0, 0],
                               suffixes=('_a', '_b'))  # 左右出现重复列名需要指定suffixes  来区别重复的列名
        df_weather2.columns = [0, 'max_tempe', 'min_tempe', 'avg_tempe', 'humidity']  # 合并后的表格每列重新命名
        '''绘制平均气温箱线图'''
        # df_weather2.boxplot(column=['avg_tempe'])
        # plt.show()

        # 将min_tempe列中所有小于-800的数据替换成min_tempe的中位数
        df_weather2['min_tempe'][df_weather2['min_tempe'] < -800] = df_weather2['min_tempe'].median()
        # 将avg_tempe列中所有小于-40的数据替换成avg_tempe的中位数
        df_weather2['avg_tempe'][df_weather2['avg_tempe'] < -40] = df_weather2['avg_tempe'].median()
        # 将humidity列中所有小于-8000的数据替换成humidity的中位数
        df_weather2['humidity'][df_weather2['humidity'] < -8000] = df_weather2['humidity'].median()
        # 将humidity列中所有大于2000的数据替换成humidity的中位数
        df_weather2['humidity'][df_weather2['humidity'] > 2000] = df_weather2['humidity'].median()

        # 简单数据分析方法
        # self.data_analy(df_weather2)
        df_load = self.df_load.drop(self.df_load.tail(1).index)
        # 绘制箱线图
        # self.boxplot_picture(df_load)
        df_data = self.nan_value_fill(df_load, df_weather2)

        return df_data

    # a = df_data.isna().sum()
    def Is_holidays(self, df_data):
        cal = calendar()
        holidays = cal.holidays(start=df_data[0].min(), end=df_data[0].max())  # 这里设置了假期的预测范围

        # 判断中国法定节假日
        def my_isholiday(s):
            if s < pd.Timestamp('2004-01-01'):  # 节假日判断库只支持2003年到当前日期的判断
                return s in holidays
            else:
                return is_holiday(s)
        # 将数据表格中的数据转换成标准日期格式并判断日期是否为的节假日，为表格添加节假日判断列
        df_data[0] = pd.to_datetime(df_data[0], format='%Y%m%d')
        df_data['type_of_day'] = df_data[0].apply(lambda s: s.dayofweek).astype('object')
        df_data['holiday'] = df_data[0].apply(my_isholiday).astype('object')
        df_data = pd.get_dummies(df_data)
        # df_data.to_excel('data.xlsx')
        return df_data

    # a = df_data.tail(5)
    # print(a)

    # 数据特征提取
    def data_features(self, df_data):
        data_norm = df_data.iloc[:, 1:].values.astype('float64', copy=False)
        data_norm[:, :96] = (data_norm[:, :96] / 7000)
        data_norm[:, 96:99] = data_norm[:, 96:99] / 20
        data_norm[:, 99] = data_norm[:, 99] / 100
        Y = data_norm[7:, :96]
        X = np.zeros((1975, 685))  # 1975 =1874-7+1,685=96*7+4+7+2 七天负荷+4个天气特征+星期7天+是否节假日亮列
        for idx in range(7, len(data_norm)):
            X[idx - 7] = np.append(data_norm[idx - 7:idx, :96], data_norm[idx, 96:])
        np.savetxt('../data/features.csv', X, delimiter=",")
        np.savetxt('../data/labels.csv', Y, delimiter=",")
        labels = torch.tensor(Y, dtype=torch.float32)
        labels = labels.to(self.device)  # 自行定义使用cpu和GPU
        features = torch.tensor(X, dtype=torch.float32)
        features = features.to(self.device)  # 自行定义使用cpu和GPU

        return labels, features

    # 定义网络
    def get_net(self, num_inputs=685, num_hiddens1=520, num_outputs=96):
        net = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens1),
            nn.ReLU(),
            nn.Linear(num_hiddens1, num_outputs)
        )
        for params in net.parameters():
            init.normal_(params, mean=0, std=0.01)
        net.to('cuda')
        return net

    # 定义网络正确率函数
    def accuary(self, y_pred, y_real):
        return 1 - np.sqrt(np.mean(((y_pred - y_real) / y_real) ** 2, axis=1))

    # 定义网络训练函数
    def train(self, net, train_features, train_labels, test_features, test_labels,
              num_epochs, learning_rate, batch_size):
        train_ls, test_ls = [], []
        train_accus, test_accus = [], []
        best_test_accu = 0
        dataset = torch.utils.data.TensorDataset(train_features, train_labels)  # 将数据转化成tensor函数
        train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)  # 将数据的按照指定batch_size处理数据
        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)  # 定义网络优化器
        loss = torch.nn.MSELoss()
        net = net.float()
        for epoch in range(num_epochs):
            net.train(True)
            for X, y in train_iter:
                l = loss(net(X.float()), y.float())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            train_ls.append(l)
            # 模型验证
            net.eval()
            bb = net(train_features)
            train_accu = self.accuary(bb.detach().cpu().numpy(), train_labels.detach().cpu().numpy())
            train_accus.append(train_accu.mean())
            if test_labels is not None:
                test_l = loss(net(test_features), test_labels)
                test_ls.append(test_l)
                test_accu = self.accuary(net(test_features).detach().cpu().numpy(), test_labels.detach().cpu().numpy())
                test_accus.append(test_accu.mean())
                if test_accu.mean() > best_test_accu:
                    torch.save(net, './model.pt')  # 保存最优模型
                    best_test_accu = test_accu.mean()
            if epoch % 10 == 0:
                print('epooch %d: train mse %.4f, test mes %.4f, train accuary %.4f, test accuary %.4f' % (
                    epoch, l, test_l, train_accu.mean(), test_accu.mean()))

        return train_ls, test_ls, train_accus, test_accus

    # 训练模型
    def model_train(self, labels, features):
        # 将数据切分成训练集和测试集
        X = np.zeros((1975, 685))
        train_index = np.arange(0, len(X) - 155)
        test_index = np.arange(len(X) - 155, len(X))
        train_feautures = features[train_index]
        train_labels = labels[train_index]
        test_feautures = features[test_index]
        test_labels = labels[test_index]
        # 构建网络训练网络
        net = self.get_net().to(self.device)
        # print(net)
        num_epochs, learning_rate, batch_size = 1000, 1e-3, 32
        train_ls, test_ls, train_accus, test_accus = self.train(net, train_feautures, train_labels, test_feautures,
                                                                test_labels,
                                                                num_epochs, learning_rate, batch_size)
        # 将tensor格式数据转换成数值数据
        train_ls = [i.item() for i in train_ls]
        test_ls = [i.item() for i in test_ls]
        train_accus = [i.item() for i in train_accus]
        test_accus = [i.item() for i in test_accus]
        best_epoch = np.array(test_ls).argmin()
        print('test best mse: %.6f' % test_ls[best_epoch])
        print('test best accuary: ', test_accus[best_epoch])
        # 绘制模型训练时参数变化图形
        # self.train_model_plot(train_ls, test_ls, train_accus, test_accus)

        # 验证模型参数图形
        net = torch.load('./model.pt')
        net.eval()
        y_pred = net(test_feautures).cpu().detach().numpy()
        y_real = test_labels.cpu().detach().numpy()
        precision = self.accuary(y_pred, y_real)
        # self.eval_plot(precision)
        # self.predicted_real_values_plot(y_pred, y_real, precision)

        return train_ls, test_ls, train_accus, test_accus, precision

    def Train(self):
        df_data = self.data_clear()
        df_data = self.Is_holidays(df_data)
        labels, features = self.data_features(df_data)
        train_ls, test_ls, train_accus, test_accus, precision = self.model_train(labels, features)

        something_wrong = np.where(precision < 0.95)[0]
        df_temp = df_data.iloc[-155:, [0, 109]]
        df_temp = df_temp.iloc[something_wrong, :]

        df_temp['idx'] = np.array(something_wrong) * 96
        df_temp['precision'] = precision[something_wrong]
        a = df_temp
        print(a)


if __name__ == '__main__':
    data_path = '../data/STLF_DATA_IN_1.xls'
    model = MODEL(data_path)
    model.Train()

# print(np.array(something_wrong) * 96)
# for print_index in something_wrong:
#     plt.figure()
#     plt.plot(y_real[:print_index].reshape((-1, 1)) * 7000, label='real')  # 原代码这个位置是y_real[print_index]  这个地方还是需要修改
#     plt.plot(y_pred[:print_index].reshape((-1, 1)) * 7000, label='pred')
#     plt.legend()
#     plt.xlabel('time')
#     plt.ylabel('load')
#     plt.grid()
#     plt.title(96 * print_index)
#     plt.show()
# #
# # if __name__ == '__main__':
# #     data_path = '../data/STLF_DATA_IN_1.xls'  # 原始数据的路径
