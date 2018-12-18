from utils import load_and_init_datasets, fill_DataserUser
from utils import make_boxplot, make_correlation
from utils import show_pca_features, show_pca_3D, show_pca_2D
from utils import get_best_Kmeans, show_Kmeans_2D
from utils import load_and_split_to_numpy
from utils import StudentPerceptron
from utils import Make_clustering
# %%
# TODO Changer le feedback et garder uniquement le type d'erreur.

# TODO Faire un dictionnaire de data par user (nb réussi, score moyen etc...)
# DONE Moyenne exercice quiz et exam et temps exam partiel et final.

# TODO Pour les quiz ou les exams le temps est limité 20mn quiz 3h exam.
#      Calculez le temps de réponse.

# TODO Calculez la moyenne sur chaque notebook pour voir si
#      il est difficile ou pas

# TODO Regarder la rétroaction du correcteur => regex
#      => classer les erreurs par un chiffre
# TODO Nombre de soumissions par notebooks (colonne count)

# TODO Faire un T-sne comme dans le devoir 5 (cf code d5q4)
# TODO Obtenir des résultats avec LinearSVC regression avec un svm
# TODO Obtenir des résultats avec le PCM fait main (quasi bon)
# TODO Classifier avec les algos vu en cours ! EM, Kmeans, SVM, PCM, Noyeau gaussien etc...
# TODO !!!!!!!!!!!!!!    FAIRE LE RAPPORT   !!!!!!!!!!!!!!!

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go


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

"""
dataset16, DatasetUser = load_and_init_datasets("./nb_entries_a16.json")
DatasetUser = fill_DataserUser(dataset16, DatasetUser)
DatasetUser.to_pickle('DatasetUser.save')
"""
DatasetUser = pd.read_pickle('DatasetUser.save')

# make_boxplot(DatasetUser.drop(columns=['Eleve']))
# correlation, maxcorr, indmaxcorr = make_correlation(DatasetUser)
# print(maxcorr, indmaxcorr)

# show_pca_features(DatasetUser.drop(columns=['Eleve']))

# show_pca_3D(DatasetUser.drop(columns=['Eleve']))
# show_pca_2D(DatasetUser.drop(columns=['Eleve']))

# best_k = get_best_Kmeans(DatasetUser.drop(columns=['Eleve']))
# print("Le meilleurs nombre de clusters est : "+str(best_k))
# show_Kmeans_2D(DatasetUser.drop(columns=['Eleve']).dropna())


# print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
X_train, y_train, X_test, y_test = load_and_split_to_numpy(
    'DatasetUser.save', 'Moyenne')
# %%
RegSVM = LinearSVC()
RegSVM.fit(X_train, y_train)
print('score', RegSVM.score(X_test, y_test))
Pcm = StudentPerceptron()
# Pcm.load()
Pcm.train(batch_size=1)
Pcm.score()
Pcm.save()
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
