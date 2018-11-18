# %%
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as st
import pandas as pd
import numpy as np
import sys
print(sys.path)
print(sys.executable)

# TODO Changer le feedback et garder uniquement le type d'erreur.
# TODO Faire un dictionnaire de data par user (nb réussi, score moyen etc...)
# TODO Pour les quiz ou les exams le temps est limité 20mn quiz 3h exam. Calculez le temps de réponse.
# TODO Calculez la moyenne sur chaque notebook pour voir si il est difficile ou pas
# TODO Regarder la rétroaction du correcteur => regex => classer les erreurs par un chiffre
# TODO
# TODO
# TODO
# TODO
# TODO
# TODO
# TODO
# TODO
# TODO

# <ul>
# <li>'user': identifiant anonyme de l'étudiant qui a soumis,</li>
# <li>'notebook': numéro du notebook de l'exercice, du quiz ou de la question d'examen,</li>
# <li>'semester': la session de la soumission (p.ex. 'A18'),</li>
# <li>'type': le type de notebook ('exercice', 'quiz' ou 'exam'),</li>
# <li>'answer': le contenu de la cellule de solution de l'étudiant,</li>
# <li>'tests': le contenu de la cellule de test,</li>
# <li>'feedback': la rétroaction du correcteur automatique,</li>
# <li>'score': la note accordée par le correcteur,</li>
# <li>'count': le numéro de la soumission,</li>
# <li>'valid': booléen indiquant si la soumission était valide ou pas,</li>
# <li>'date': la date de la soumission (str de l'objet datetime, p.ex. '2018-10-15 15:14:55.788330'),</li>
# </ul>


# %%
dataset16 = pd.read_json("./nb_entries_a16.json")
dataset16

# %%
dataset16.replace({'quiz': 1, 'exercise': 2, 'exam': 3}, inplace=True)
dataset16
# %%
DatasetUser = pd.DataFrame(index=range(len(set(dataset16['user']))))
DatasetUser['Eleve'] = pd.Series(
    sorted(set(dataset16['user'])), index=DatasetUser.index)  # %%
DatasetUser['Moyenne'] = pd.Series(index=DatasetUser.index)
DatasetUser['Notebookfaits'] = pd.Series(index=DatasetUser.index)

# %%
for i in DatasetUser.index:
    usr = DatasetUser.iloc[i]['Eleve']
    results = dataset16.loc[(dataset16['user'] == usr)]['score'].value_counts()
    if usr == '1bV8Dm':
        print(results)
        print("nombre notebook fait", results.sum())
        print(results.describe())
        print("moy", np.average(results.index, weights=results.values))

    DatasetUser.loc[i, 'Moyenne'] = np.average(
        results.index, weights=results.values)

    DatasetUser.loc[i, 'Notebookfaits'] = results.sum()

# %%
DatasetUser


# %%
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
####################   COMMANDES UTILES  ##########################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
"""
dataset16[['notebook', 'date']].where(dataset16['user'] == 'HhwFBj').dropna()
dataset16['feedback'][0]
set(dataset16['type'])
hasPassed100 = dataset16['feedback'].str.contains('Bravo!', na=np.nan, case=True)
indreussi100 = hasPassed[hasPassed100].index
hasPassed = dataset16['user'].where(dataset16['score']>50)
columns_to_numbers = dict(zip(dataset16.columns, range(len(dataset16.columns))))
dataset16.where(dataset16['score'] == 100).dropna()
DatasetUser.loc[3, 'Eleve']



"""
# %%
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################

# %%
plt.figure(figsize=[10, 5])
dataset16.rename(columns=columns_to_numbers).boxplot()
plt.show()


# %%
correlations = dataset16.rename(columns=columns_to_numbers).corr()


# %%
names = list(correlations.columns)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 46, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# %%
maximum = [0, 0, 0]
ind_max = [[0, 0], [0, 0], [0, 0]]
for i in correlations.columns:
    for j in correlations.index:
        # We look only bellow the diagonal of the correlation matrix (which is symetric)
        if i < j:
            if abs(correlations[i][j]) >= min(maximum):
                ind_max[maximum.index(min(maximum))] = [i, j]
                maximum[maximum.index(min(maximum))] = abs(correlations[i][j])
print(maximum)
print(ind_max)


# %%
def get_column_name(nb, dic):
    for i in dic:
        if dic[i] == nb:
            return i


# %%
for i in range(len(ind_max)):
    print("Pair_"+str(i))
    print(get_column_name(ind_max[i][0], columns_to_numbers))
    print(get_column_name(ind_max[i][1], columns_to_numbers))
    print()


# %%
dataset_std = pd.DataFrame(StandardScaler().fit_transform(
    dataset16.drop(labels=dataset16.columns[0], axis=1)))
cov_std = dataset_std.corr()
eig_vals, eig_vect = np.linalg.eig(cov_std)
eig_pairs = [(np.abs(eig_vals[i]), eig_vect[:, i])
             for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
sum_ev = sum(eig_vals)
pve = [(i / sum_ev)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_pve = np.cumsum(pve)

fig = plt.figure(figsize=[10, 5])
plt.scatter(['PC%s' % i for i in range(len(dataset_std.columns))], pve, s=80)
plt.scatter(['PC%s' % i for i in range(len(dataset_std.columns))],
            cum_var_pve, marker='+')
plt.legend(['Variance', 'Cumulative variance'])
plt.show()


# %%
pca = PCA().fit(dataset16.drop(labels=dataset16.columns[0], axis=1))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative variance')


# %%
dataset_pca = PCA(n_components=2).fit_transform(dataset_std)


# %%
pca = PCA(n_components=3).fit(dataset16.drop(
    labels=dataset16.columns[0], axis=1))
X_reduced = pca.transform(dataset16.drop(labels=dataset16.columns[0], axis=1))
trace1 = go.Scatter3d(
    x=X_reduced[:, 0],
    y=X_reduced[:, 1],
    z=X_reduced[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color='blue',
        opacity=1
    )

)

dc_1 = go.Scatter3d(x=[0, pca.components_.T[0][0]],
                    y=[0, pca.components_.T[0][1]],
                    z=[0, pca.components_.T[0][2]],
                    marker=dict(size=1,
                                color="rgb(84,48,5)"),
                    line=dict(color="red",
                              width=6),
                    name="Var1"
                    )
dc_2 = go.Scatter3d(x=[0, pca.components_.T[1][0]],
                    y=[0, pca.components_.T[1][1]],
                    z=[0, pca.components_.T[1][2]],
                    marker=dict(size=1,
                                color="rgb(84,48,5)"),
                    line=dict(color="green",
                              width=6),
                    name="Var2"
                    )
dc_3 = go.Scatter3d(x=[0, pca.components_.T[2][0]],
                    y=[0, pca.components_.T[2][1]],
                    z=[0, pca.components_.T[2][2]],
                    marker=dict(size=1,
                                color="rgb(84,48,5)"),
                    line=dict(color="blue",
                              width=6),
                    name="Var3"
                    )
dc_4 = go.Scatter3d(x=[0, pca.components_.T[3][0]],
                    y=[0, pca.components_.T[3][1]],
                    z=[0, pca.components_.T[3][2]],
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
plot(fig, filename='3d-scatter-tupac-with-mac')


# %%
dat = pd.DataFrame(StandardScaler().fit_transform(dataset16.drop(
    labels=dataset16.columns[0], axis=1).rename(columns=columns_to_numbers)))

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


# %%
k = range(2, 21)
silhouette = [0.0]*21
for n_clusters in k:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(dataset_pca)
    silhouette_avg = silhouette_score(dataset_pca, cluster_labels)
    silhouette[n_clusters] = silhouette_avg
    print(str(n_clusters) + " : " + str(silhouette_avg))
# We compute the score for each cluster and take the closest to 1
best_nb_clust = silhouette.index(max(silhouette))
print("The best number of cluster is : "+str(best_nb_clust))


# %%
kmeans_label = KMeans(n_clusters=2, random_state=10).fit_predict(dataset_pca)
plt.scatter(dataset_pca[:, 0], dataset_pca[:, 1],
            c=kmeans_label, s=50, cmap='viridis')


# %%
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
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

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

    plt.show()
