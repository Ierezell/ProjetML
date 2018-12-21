from utils import load_and_init_datasets, fill_DataserUser
from utils import make_boxplot, make_correlation
from utils import show_pca_features, show_pca_3D, show_pca_2D, show_tsne_3D
from utils import get_best_Kmeans, show_Kmeans_2D
from utils import load_and_split_to_numpy
from utils import StudentPerceptron
from utils import Make_clustering
from utils import test_many_classifiers

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid


# %%
# TODO Changer le feedback et garder uniquement le type d'erreur.
# Ex : <balise html> votre score blabla votre erreur (ERREUR) à garder !

# TODO Faire un dictionnaire de data par user (nb réussi, score moyen etc...)
# DONE Moyenne exercice quiz et exam et temps exam partiel et final.
# MAIS AU PLUS ON EN A AU MIEUX CEST ALORS FAISONS PLUS

# TODO Pour les quiz ou les exams le temps est limité 20mn quiz 3h exam.
#      Calculez le temps de réponse.
# Done pour exams

# TODO Calculez la moyenne sur chaque notebook pour voir si
#      il est difficile ou pas et ponderer ?

# TODO Regarder la rétroaction du correcteur => regex
#      => classer les erreurs par un chiffre => stat sur les erreurs

# TODO Nombre de soumissions par notebooks (colonne count)

# TODO Faire un T-sne comme dans le devoir 5 (cf code d5q4)
# #Done

# TODO Obtenir des résultats avec LinearSVC regression avec un svm
# TODO Obtenir des résultats avec le PCM fait main (quasi bon StudentNet)
# TODO Classifier avec les algos vu en cours !
#  EM,
#  Kmeans  (done),
#  SVM (almost done, utiliser LinearSVC et checker la perf),
#  PCM (almost done),
#  Noyeau gaussien (Done)
#  etc...
# TODO Avec ça, faire un classificateur par vote
# TODO !!!!!!!!!!!!!!    FAIRE LE RAPPORT   !!!!!!!!!!!!!!!


# <ul>
# <li>'user': identifiant anonyme de l'étudiant qui a soumis,</li>
# <li>'notebook': numéro du notebook de l'exercice, du quiz ou de la question
#        d'examen,</li>
# <li>'semester': la session de la soumission (p.ex. 'A18'),</li>
# <li>'type': le type de notebook ('exercice', 'quiz' ou 'exam'),</li>
# <li>'answer': le contenu de la cellule de solution de l'étudiant,</li>
# <li>'tests': le contenu de la cellule de test,</li>
# <li>'feedback': la rétroaction du correcteur automatique,</li>
# <li>'score': la note accordée par le correcteur,</li>
# <li>'count': le numéro de la soumission,</li>
# <li>'valid': booléen indiquant si la soumission était valide ou pas,</li>
# <li>'date': la date de la soumission (str de l'objet datetime, p.ex.
#                        '2018-10-15 15:14:55.788330'),</li>
# </ul>

# %%
# dataset16, DatasetUser = load_and_init_datasets("./nb_entries_a16.json")
# DatasetUser = fill_DataserUser(dataset16, DatasetUser)
# DatasetUser.to_pickle('DatasetUser.save')

# %%
print("------------------------------------------------------")
print("Lecture des données enregistrées")
DatasetUser = pd.read_pickle('DatasetUser.save')


# %%

print("------------------------------------------------------")
print("Affichage Boxplot de toutes des différentes données")
make_boxplot(DatasetUser.drop(columns=['Eleve']))

print("------------------------------------------------------")
print("Calculs et affichage des covariances entre les différentes données")
correlation, maxcorr, indmaxcorr = make_correlation(DatasetUser)

print(f"Les {len(maxcorr)} meilleurs covariances sont :")
for i in range(len(maxcorr)):
    print(f"Entre {indmaxcorr[i][0] :<14} et {indmaxcorr[i][1] :<14} : {maxcorr[i]}")

print("------------------------------------------------------")
print("Affichage PCA Features")
show_pca_features(DatasetUser.drop(columns=['Eleve']))

print("Affichage PCA 3D")
show_pca_3D(DatasetUser.drop(columns=['Eleve']))

print("Affichage PCA 2D")
show_pca_2D(DatasetUser.drop(columns=['Eleve']))

print("------------------------------------------------------")
print("Calcul du meilleur nombre de clusters")
best_k = get_best_Kmeans(DatasetUser.drop(columns=['Eleve']))

print(f"Le meilleur nombre de clusters est : {best_k}")

print("Affichage du K Means 2D")
show_Kmeans_2D(DatasetUser.drop(columns=['Eleve']).dropna())

print("------------------------------------------------------")
print("Make Cluster")
Make_clustering(DatasetUser, 'Moyenne')
show_tsne_3D(DatasetUser.drop(columns=['Eleve']))


print("------------------------------------------------------")
print("Entrainement des classifieurs")

X, y, X_train, y_train, X_test, y_test = load_and_split_to_numpy(
    'DatasetUser.save', 'Moyenne')

classifieurs = [LinearSVC(),
                StudentPerceptron(),
                # QuadraticDiscriminantAnalysis(),
                LinearDiscriminantAnalysis(),
                GaussianNB(),
                NearestCentroid()]

print(f"Il y a {len(classifieurs)} classifieurs :")
for i in range(len(classifieurs)):
    print(f"{i :<2}: {type(classifieurs[i])}")

print("Calcul des erreurs sur tous les classifieurs")
ErrorsClassifieurs = test_many_classifiers(X, y, classifieurs, Kfold=5)

for i in range(len(classifieurs)):
    print(f"{i :<2}: {str(type(classifieurs[i])) :<66} : {ErrorsClassifieurs[i]}")

# Pcm.load()
# Pcm.train(batch_size=1)
# Pcm.score()
# Pcm.save()
# %%

# TEST POUR LE TEMPS QUIZZ NON FONCTIONNEL !
""" tempsQuiz = dataset16.loc[(dataset16['user'] == 'HhwFBj') &
                          (dataset16['type'] == 2) &
                          (dataset16['valid'] == True)][
    ['notebook', 'count', 'date']]
tempsQuiz

for n in numNotebookQuiz:
    tps = tempsQuiz.where(tempsQuiz['notebook'] == n).dropna()
    DatasetUser.loc['TempsQuizz_{}'.format(n)] = (tps.max()-tps.min()).seconds
tps = tempsQuiz[['date', 'count', 'notebook']].where(
    tempsQuiz['notebook'] == 44).dropna()['date']
(tps.max()-tps.min()).seconds """
