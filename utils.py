
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import time
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import plot, iplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
# import cufflinks as cf
# cf.go_offline()


def load_and_init_datasets(path):
    dataset16 = pd.read_json("./nb_entries_a16.json")
    dataset16.replace({'exercise': 1, 'quiz': 2, 'exam': 3}, inplace=True)

    DatasetUser = pd.DataFrame(index=range(len(set(dataset16['user']))))

    DatasetUser['Eleve'] = pd.Series(
        sorted(set(dataset16['user'])), index=DatasetUser.index)  # %%
    DatasetUser['Moyenne'] = pd.Series(index=DatasetUser.index)
    DatasetUser['MoyenneExec'] = pd.Series(index=DatasetUser.index)
    DatasetUser['MoyenneQuiz'] = pd.Series(index=DatasetUser.index)
    DatasetUser['MoyenneExam'] = pd.Series(index=DatasetUser.index)
    DatasetUser['Notebookfaits'] = pd.Series(index=DatasetUser.index)
    DatasetUser['TempsExam1'] = pd.Series(index=DatasetUser.index)
    DatasetUser['TempsExam2'] = pd.Series(index=DatasetUser.index)
    # DatasetUser['TempsQuiz'] = pd.Series(index=DatasetUser.index)
    # for n in numNotebookQuiz:
    #     DatasetUser['TempsQuizz_{}'.format(n)] = pd.Series(index=DatasetUser.index)
    # numNotebookQuiz = set(dataset16['notebook'].loc[(dataset16['type'] == 2) &
    #                                                 (dataset16['valid'] == True)])
    return dataset16, DatasetUser


def fill_DataserUser(dataset, DatasetUser):
    for i in DatasetUser.index:
        usr = DatasetUser.iloc[i]['Eleve']

        results = dataset.loc[(dataset['user'] == usr)][['type', 'score']]

        resultsTot = results['score'].value_counts()

        resultExec = results['score'].where(
            results['type'] == 1).value_counts()

        resultQuiz = results['score'].where(
            results['type'] == 2).value_counts()

        resultExam = results['score'].where(
            results['type'] == 3).value_counts()

        DatasetUser.loc[i, 'Moyenne'] = np.average(
            resultsTot.index, weights=resultsTot.values)

        if 1 in set(results['type']):
            DatasetUser.loc[i, 'MoyenneExec'] = np.average(
                resultExec.index, weights=resultExec.values)

        if 2 in set(results['type']):
            DatasetUser.loc[i, 'MoyenneQuiz'] = np.average(
                resultQuiz.index, weights=resultQuiz.values)

        if 3 in set(results['type']):
            DatasetUser.loc[i, 'MoyenneExam'] = np.average(
                resultExam.index, weights=resultExam.values)

        DatasetUser.loc[i, 'Notebookfaits'] = resultsTot.sum()

        tempsQuiz = dataset.loc[(dataset['user'] == usr) &
                                (dataset['type'] == 2) &
                                (dataset['valid'] == True)][
            ['notebook', 'count', 'date']]

        tempsExam = dataset.loc[(dataset['user'] == usr) &
                                (dataset['type'] == 3) &
                                (dataset['valid'] == True)][
            ['notebook', 'count', 'date']]
        if not tempsExam.empty:
            tempsExam1 = tempsExam.where(
                tempsExam['date'].dt.date == tempsExam['date'].loc[
                    tempsExam.index[0]].date()).dropna()['date']

            tempsExam2 = tempsExam.where(
                tempsExam['date'].dt.date == tempsExam['date'].loc[
                    tempsExam.index[-1]].date()).dropna()['date']

            DatasetUser.loc[i, 'TempsExam1'] = (
                tempsExam1.max()-tempsExam1.min()).seconds

            DatasetUser.loc[i, 'TempsExam2'] = (
                tempsExam2.max()-tempsExam2.min()).seconds
    return DatasetUser


def make_boxplot(dataset):
    plt.figure(figsize=[10, 5])
    normalized_Dataset = (dataset-dataset.mean())/dataset.std()
    normalized_Dataset.boxplot()
    plt.show()


def make_correlation(dataset):
    correlations = dataset.corr()
    names = list(correlations.columns)
    fig = plt.figure(figsize=[10, 15])
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(correlations), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
    maximum = [0, 0, 0]
    ind_max = [[0, 0], [0, 0], [0, 0]]
    for i in correlations.columns:
        for j in correlations.index:
            # We look only bellow the diagonal of
            # the correlation matrix (which is symetric)
            if i < j:
                if abs(correlations[i][j]) >= min(maximum):
                    ind_max[maximum.index(min(maximum))] = [i, j]
                    maximum[maximum.index(min(maximum))] = abs(
                        correlations[i][j])
    return correlations, maximum, ind_max


def show_pca_features(dataset):
    DatasetUser_std = pd.DataFrame(
        StandardScaler().fit_transform(dataset.dropna()))
    cov_std = DatasetUser_std.corr()
    eig_vals, eig_vect = np.linalg.eig(cov_std)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vect[:, i])
                 for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    sum_ev = sum(eig_vals)
    pve = [(i / sum_ev)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_pve = np.cumsum(pve)

    fig = plt.figure(figsize=[10, 5])
    plt.scatter(['PC%s' % i for i in range(
        len(DatasetUser_std.columns))], pve, s=80)
    plt.scatter(['PC%s' % i for i in range(len(DatasetUser_std.columns))],
                cum_var_pve, marker='+')
    plt.legend(['Variance', 'Cumulative variance'])
    plt.show()
    """
    FAIS LA MEME CHOSE QU'AU DESSUS GRACE A LA FONCTION PCA
    pca = PCA().fit(DatasetUser.drop(columns=['Eleve']).dropna())
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative variance')
    plt.show()
    """


def show_pca_3D(dataset):
    pca = PCA(n_components=3).fit(dataset.dropna())
    X_reduced = pca.transform(dataset.dropna())
    trace1 = go.Scatter3d(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        z=X_reduced[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            opacity=0.8
        )

    )

    dc_1 = go.Scatter3d(x=[0, 30*pca.components_.T[0][0]],
                        y=[0, 30*pca.components_.T[0][1]],
                        z=[0, 30*pca.components_.T[0][2]],
                        marker=dict(size=1,
                                    color="rgb(84,48,5)"),
                        line=dict(color="red",
                                  width=6),
                        name="Var1"
                        )
    dc_2 = go.Scatter3d(x=[0, 30*pca.components_.T[1][0]],
                        y=[0, 30*pca.components_.T[1][1]],
                        z=[0, 30*pca.components_.T[1][2]],
                        marker=dict(size=1,
                                    color="rgb(84,48,5)"),
                        line=dict(color="green",
                                  width=6),
                        name="Var2"
                        )
    dc_3 = go.Scatter3d(x=[0, 30*pca.components_.T[2][0]],
                        y=[0, 30*pca.components_.T[2][1]],
                        z=[0, 30*pca.components_.T[2][2]],
                        marker=dict(size=1,
                                    color="rgb(84,48,5)"),
                        line=dict(color="pink",
                                  width=6),
                        name="Var3"
                        )
    dc_4 = go.Scatter3d(x=[0, 30*pca.components_.T[3][0]],
                        y=[0, 30*pca.components_.T[3][1]],
                        z=[0, 30*pca.components_.T[3][2]],
                        marker=dict(size=1,
                                    color="rgb(84,48,5)"),
                        line=dict(color="yellow",
                                  width=6),
                        name="Var4"
                        )

    data = [trace1, dc_1, dc_2, dc_3, dc_4]
    layout = go.Layout(
        xaxis=dict(
            title='PC1',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=data, layout=layout)
    # iplot(fig, filename='pca_3D')
    try:
        plot(fig, filename='PCA_3D.html')
    except TypeError:
        pass


def show_pca_2D(dataset):
    dat = pd.DataFrame(StandardScaler().fit_transform(dataset.dropna()))
    pca = PCA(n_components=2)
    pca.fit(dat)
    xvector = pca.components_[0]
    yvector = pca.components_[1]
    xs = pca.transform(dat)[:, 0]
    ys = pca.transform(dat)[:, 1]
    for i in range(len(xvector)):
        plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                  color='r', width=0.0005, head_width=0.0025)
        plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
                 list(dat.columns.values)[i], color='r')
    for i in range(len(xs)):
        plt.plot(xs[i], ys[i], 'bo', alpha=0.5)
    plt.title('Biplot')
    plt.show()


def get_best_Kmeans(dataset):
    k = range(2, 21)
    silhouette = [0.0]*21
    for n_clusters in k:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(dataset.dropna())
        silhouette_avg = silhouette_score(dataset.dropna(), cluster_labels)
        silhouette[n_clusters] = silhouette_avg
        print(str(n_clusters) + " : " + str(silhouette_avg))
    # We compute the score for each cluster and take the closest to 1
    best_nb_clust = silhouette.index(max(silhouette))
    return best_nb_clust


def show_Kmeans_2D(dataset):
    dataset_std = pd.DataFrame(
        StandardScaler().fit_transform(dataset.dropna()))

    dataset_pca = PCA(n_components=2).fit_transform(dataset_std)

    kmeans_label = KMeans(
        n_clusters=2, random_state=10).fit_predict(dataset_pca)

    plt.scatter(dataset_pca[:, 0], dataset_pca[:, 1],
                c=kmeans_label, s=50, cmap='viridis')

    X = dataset_pca
    range_n_clusters = range(2, 21)

    for n_clusters in range_n_clusters:
        # Subplot : 1 row ,2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # The 1st subplot is the silhouette plot
        fig.set_size_inches(18, 7)

        # Limit of the figure for the silhouette -1, 1
        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for blank space between silhouette
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # Silhouette score between -1 (worse) and 1 (better)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.inferno(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = plt.cm.inferno(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=cluster_labels, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        # plt.show()


def load_and_split_to_numpy(path, column_label):
    datas = pd.read_pickle(path)
    datas = datas.drop(columns=['Eleve']).dropna()
    y = datas[column_label].values
    X = datas.drop(columns=[column_label]).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=666, shuffle=True)
    return X_train, y_train, X_test, y_test


class _StudentNet(nn.Module):
    def __init__(self, input_size):
        super(_StudentNet, self).__init__()

        self.L1 = nn.Linear(input_size, input_size*2)
        self.L2 = nn.Linear(input_size*2, input_size)
        self.L3 = nn.Linear(input_size, input_size//2)
        self.L4 = nn.Linear(input_size//2, 1)
        self.drop = nn.Dropout(p=0.6)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        y = F.relu(self.L1(x), inplace=True)
        y = self.drop(y)
        y = F.relu(self.L2(y), inplace=True)
        y = self.drop(y)
        y = F.relu(self.L3(y), inplace=True)
        y = self.Sig(self.L4(y))
        # y = y.mean(dim=2)
        return y


class _StudentDataset(Dataset):
    def __init__(self, datalocation, column_to_guess):
        super(_StudentDataset, self).__init__()
        loadedPanda = pd.read_pickle(datalocation)
        self._datas = loadedPanda.drop(columns=['Eleve']).dropna()
        self.y = self._datas[column_to_guess].values
        self.X = self._datas.drop(columns=[column_to_guess]).values

    def __getitem__(self, index):
        return Variable(torch.FloatTensor([self.X[index]]), requires_grad=True), Variable(torch.FloatTensor([self.y[index]/100]), requires_grad=False)

    def __len__(self):
        return len(self._datas)


class StudentPerceptron():
    def __init__(self, column_to_guess='Moyenne'):
        self.dataset = _StudentDataset('./DatasetUser.save', column_to_guess)
        self.model = _StudentNet(self.dataset[0][0].size()[1])

    def train(self, batch_size=32, epoch=50, lr=0.01, momentum=0.7):

        self._indices = list(range(len(self.dataset)))
        self._train_idx = np.random.choice(self._indices, size=int(
            np.floor(0.6*len(self._indices))), replace=False)

        self._test_idx = list(set(self._indices) - set(self._train_idx))

        self._train_sampler = SubsetRandomSampler(self._train_idx)
        self._test_sampler = SubsetRandomSampler(self._test_idx)
        self.train_loader =\
            torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                        sampler=self._train_sampler)
        self.test_loader =\
            torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                        sampler=self._test_sampler)

        optimizer = Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='sum')
        for i_epoch in range(epoch):
            start_time, train_losses = time.time(), []
            for b in self.train_loader:
                pred, targets = b
                pred = pred.view(len(pred), self.dataset[0][0].size()[1])
                targets = targets.view(len(pred))
                #print("pred", pred)

                optimizer.zero_grad()
                predictions = self.model(pred)
                predictions = predictions.view(len(pred))
                print("predictions", predictions.data[0],
                      "--- Target", targets.data[0])
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
                i_epoch+1, epoch, np.mean(train_losses),
                time.time()-start_time))

    def score(self):
        accuracy = 0
        for i, b in enumerate(self.test_loader):
            pred, targets = b
            pred = Variable(torch.FloatTensor(pred), requires_grad=False)
            predictions = self.model.forward(pred)
            accuracy += torch.where(targets+0.1 > predictions > targets-0.1,
                                    torch.tensor([1.0]),
                                    torch.tensor([0.0]))
        print("Score Pcm : ", accuracy/i)
        return accuracy/i

    def save(self, path='./Backup/StudentNet.pt'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='./Backup/StudentNet.pt'):
        self.model.load_state_dict(torch.load(path))
