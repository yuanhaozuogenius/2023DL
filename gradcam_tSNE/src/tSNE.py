import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import decomposition, manifold
from sklearn.preprocessing import StandardScaler


class tSNE:
    @staticmethod
    def load_data():
        print("*****load_data start******")
        data = pd.read_csv('data/train.csv')
        data = data.head(1000)
        print(data.head())
        df_labels = data.label
        # print(df_labels, type(df_labels))
        df_data = data.drop('label', axis=1)
        # Count plot for the labels
        sns.countplot(data=data, x='label')
        plt.title("load_data points")
        plt.show()  # 显示图像
        plt.clf()  # 清除图像
        plt.close()  # 关闭图像

        df_data = df_data.head(1000)
        df_labels = df_labels.head(1000)

        '''scaler.fit()是用于训练Scaler的，它计算了训练数据集的均值和方差，以便在后续将其应用于测试集或新数据时进行归一化处理。
        scaler.transform()是用于对数据集进行归一化处理的，它使用先前计算的均值和方差对数据进行缩放。
        因此，我们在训练集上使用scaler.fit()来计算均值和方差，然后在训练集和测试集上分别使用scaler.transform()来对数据进行归一化处理。
        这样做是为了避免训练集和测试集之间的偏差。可以用fit_transform()一步完成'''
        # pixel_df = StandardScaler().fit_transform(df_data)
        # df_data中的数据进行标准化处理，并返回一个新的numpy.ndarray对象pixel_df。其中，StandardScaler()是一个sklearn库中的标准化工具，通过fit_transform()方法将df_data标准化。
        scaler = StandardScaler()
        # 归一化处理是指将数据按比例缩放，使之落入一个特定的范围，常见的范围是[0,1]，即使得数据的均值为0，标准差为1。
        scaler.fit(df_data)
        pixel_df = scaler.transform(df_data)

        print("pixel_df.shape：", pixel_df.shape)
        print("head 10 of tuple is :", pixel_df[:10])
        sample_data = pixel_df
        print("df_labels.shape:", df_labels.shape)
        print("sample_data.shape:", sample_data.shape)
        return sample_data, df_labels

    def origin_plot(data, label):
        print("*****origin_plot start******")
        print(type(data), data)
        data = np.column_stack((data, label))
        data = pd.DataFrame(data)
        print(type(data), data.head(10))
        # creating a new data frame for plotting of data points
        plt.scatter(data)
        plt.show()
        plt.clf()
        plt.close()

    # Implementation of tSNE
    def tSNE_method(data, label):
        print("*****tSNE_method start******")
        tsne = manifold.TSNE(n_components=2, random_state=42, verbose=2, n_iter=2000)
        '''KL散度（Kullback-Leibler Divergence），也称为相对熵（Relative Entropy），是用于比较两个概率分布差异的一种方法。
        在t-SNE算法中，KL散度被用来衡量原始高维空间中数据点之间的相似度和降维后低维空间中数据点之间的相似度之间的差异。
        t-SNE的目标就是使得低维空间中的相似度尽可能地符合高维空间中的相似度'''
        transformed_data = tsne.fit_transform(data)
        print(transformed_data.shape)
        print(data.shape)
        # Creation of new dataframe for plotting of data points
        tsne_df = pd.DataFrame(
            np.column_stack((transformed_data, label)), columns=['x', 'y', 'labels'])
        # 将tsne_df中labels列的数据类型转换为整型，并将结果分配回tsne_df中的labels列
        tsne_df.loc[:, 'labels'] = tsne_df.labels.astype(int)
        print(tsne_df.head(10))

        grid = sns.FacetGrid(tsne_df, hue='labels', height=8)
        # 使用map函数绘制散点图，并将其传递给plt.scatter，这将在grid上绘制多个子图，每个子图代表一个标签。
        # 最后，使用add_legend函数添加一个图例，以显示每个类别的颜色。
        grid.map(plt.scatter, 'x', 'y').add_legend()
        plt.title("tSNE:plot tSNE_data points")
        plt.show()


# Implementation of PCA

class PCA:
    def PCA_method(data, label):
        print("*****PCA_method start******")
        # 尝试将数据降到 2 维，但是数据集的样本数和特征数均小于 2，因此无法执行降维操作。
        pca = decomposition.PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(data)
        print("shape of pca_reduced.shape = ", pca_data.shape)

        # attaching the label for each 2-d data point
        pca_data = np.column_stack((pca_data, label))

        # creating a new data frame for plotting of data points
        pca_df = pd.DataFrame(data=pca_data, columns=("X", "Y", "labels"))
        print(pca_df.head(10))
        '''具体来说，这段代码先创建了一个 FacetGrid 对象，其输入参数是 pca_df 数据集， 
        hue 参数指定了按照 labels 特征进行分类绘图，height 参数指定了子图的高度为6。
        然后，使用 map 函数绘制了每个子图上的散点图，'X' 和 'Y' 指定了X轴和Y轴上所用的特征，这里是使用了PCA降维后的2个特征。
        最后，使用 add_legend() 添加图例。'''
        sns.FacetGrid(pca_df, hue="labels", height=6).map(plt.scatter, 'X', 'Y').add_legend()
        plt.title("PCA:plot pca_data points")
        plt.show()
        plt.clf()
        plt.close()
