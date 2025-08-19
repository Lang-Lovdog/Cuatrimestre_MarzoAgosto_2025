## Importation des bibliothèques
### Mathématiques
import pandas as pd # type: ignore
import numpy as np # type: ignore
### Visualisation
#import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
### Machine learning
#### Selection processus
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
#### Algorithmes de classification
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
#### Métriques
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import f1_score # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay # type: ignore
### Temps (pour random_state)
import time # type: ignore

import joblib # type: ignore


_ModelSelectionClassesNames_ : list =[]
_ModelSelectionClassesObjects_ : list =[]

class ModelSelection:

    def __init__(self, name, csv=None):
        if name in _ModelSelectionClassesNames_:
            raise ValueError(f"Object '{name}' already exists")
            return
        if csv is not None:
            self.loadDataset(csv)
        _ModelSelectionClassesNames_.append(name)
        _ModelSelectionClassesObjects_.append(self)
        self.name=name
        self.data   = None
        self.target = None
        self.gridParameters = {
            'SVC': {
                # Regularization
                'clf__C': [ 0.001, 0.01, 0.1, 1, 10, 20 ],
                'clf__kernel': [ 'poly', 'sigmoid', 'rbf' ],
                'clf__gamma': [ 'scale', 'auto' ],
            },
            'KNeighbors': {
                'clf__n_neighbors': [12, 24, 48, 96 ],
                'clf__weights': [ 'uniform', 'distance' ],
            },
            'RandomForest': {
                'clf__min_samples_split': [2, 4, 8, 16],
                # 2, 4, 8 and 16 estimators for each feature
                'clf__n_estimators': [72, 144, 288, 576],
                # To be fair, depth can cause unwanted complexity,
                ## The best depth should be between 3 and 27
                'clf__max_depth': [ 3, 9, 27 ],
            },
            'LogisticRegression': {
                # Regularization
                'clf__C': [ 0.001, 0.01, 0.1, 1, 10, 20 ],
                'clf__penalty': [ 'elasticnet', 'l2' ],
                # Saga is the only solver that can handle elasticnet
                'clf__solver': [ 'saga' ],
            },
        }
        self.classifiers = {
            'SVC':                SVC()                            ,
            'KNeighbors':         KNeighborsClassifier()           ,
            'RandomForest':       RandomForestClassifier()         ,
            'LogisticRegression': LogisticRegression(max_iter=850) ,
        }

        self.trainingData = {
            "x": None,
            "y": None
        }
        self.testData = {
            "x": None,
            "y": None
        }
        self.summary = []
        self.testSummary = []

    def loadDataset(self, dataset, targetName="target", test_size=0.2):
        if type(dataset) is str:
            df = pd.read_csv(dataset)
        elif type(dataset) is pd.DataFrame:
            df = dataset
        self.data = df.drop(targetName, axis=1)
        self.target = df[targetName]
        self.trainingData['x'], self.testData['x'],  \
        self.trainingData['y'], self.testData['y'] = \
            train_test_split(
                self.data, self.target,
                test_size=test_size,
                random_state=np.floor(time.time()).astype(int)%4294967296
            )

    @staticmethod
    def setGridParametersInto(modelName, gridParameters):
        _ModelSelectionClassesObjects_[_ModelSelectionClassesNames_.index(modelName)] \
            .gridParameters = gridParameters

    def setGridParameters(self, gridParameters):
        self.gridParameters = gridParameters

    def selectModels(self):
        for tag, algorithm in self.classifiers.items():
            # Prepare pipeline for grid
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', algorithm)
            ])
            
            # SetUp grid
            modelsGrid = GridSearchCV(
                pipe,
                param_grid=self.gridParameters[tag],
                scoring={
                    "accuracy": "accuracy",
                    "f1_macro": "f1_macro"
                },
                refit="accuracy",
                cv=5,
                n_jobs=-1,
                verbose=0
            )

            # Run grid test
            modelsGrid.fit(self.trainingData['x'], self.trainingData['y'])

            self.summary.append({
                "tag": tag,
                "best_accuracy":  modelsGrid.best_score_                                                ,
                "best_f1_macro":  modelsGrid.cv_results_['mean_test_f1_macro'][modelsGrid.best_index_]  ,
                "best_model":     modelsGrid.best_params_                                               ,
                "best_estimator": modelsGrid.best_estimator_                                            ,
                "score": modelsGrid.score(self.testData['x'], self.testData['y'])
            })

        self.summary = sorted(self.summary, key=lambda x: x["best_accuracy"], reverse=True)

    def testModels(self):
        for estimator in self.summary:
            model = estimator["best_estimator"]
            y_pred = model.predict(self.testData['x'])
            test_accuracy = accuracy_score(self.testData['y'], y_pred)
            f1_test = f1_score(self.testData['y'], y_pred, average="macro")
            self.testSummary.append({
                "tag": estimator["tag"],
                "accuracy": test_accuracy,
                "f1_macro": f1_test,
                "confusion_matrix": confusion_matrix(self.testData['y'], y_pred)
            })
        self.testSummary = sorted(self.testSummary, key=lambda x: x["accuracy"], reverse=True)
        pd.DataFrame(self.testSummary)

    def saveConfusionMatrix(self):
        for estimator in self.testSummary:
            confusionMatrixPlot = ConfusionMatrixDisplay(estimator["confusion_matrix"])
            confusionMatrixPlot.plot(cmap=plt.cm.magma)
            plt.title(f"Confusion Matrix for {estimator['tag']}")
            plt.savefig(f"img/ConfusionMatrices/CM_{estimator['tag']}.png")

    def getSummary(self):
        pd.DataFrame(self.summary).to_csv(f"summary_{self.name}.csv", index=False)
        return pd.DataFrame(self.summary)

    def printSummary(self):
        print(self.getSummary())

    def getBestModel(self):
        return self.summary[0]["best_estimator"]

    def printBestModel(self):
        print(self.getBestModel())

    def saveModels(self, path):
        for estimator in self.summary:
            joblib.dump(estimator["best_estimator"], f"{path}/{estimator['tag']}.joblib")

class dimensionalityReduction:
    def __init__(
        self, models,  ## Models
        dataset,       ## Dataset 
         ## EDAs parameters
        eID, nGenerations, pSize,
        ssSizeRange, eliteFraction, mRate, 
        classifierName,
        randomState=np.floor(time.time()).astype(int)%4294967296,
         ## EDAs parameters
        # Dataset Class Column
        targetName="target"
    ):
        self.loadDataset(dataset, targetName)
        self.models = models if type(models) is list else [models]
        self.EDAS_Config = {
            "experiment_id": eID,
            "num_generations": nGenerations,
            "population_size": pSize,
            "subset_size_range": ssSizeRange,
            "elite_fraction": eliteFraction,
            "mutation_rate": mRate,
            "random_state": randomState,
            "classifier_name": classifierName
        }
        pass

    def loadDataset(self, dataset, targetName="target", test_size=0.2):
        if type(dataset) is str:
            df = pd.read_csv(dataset)
        elif type(dataset) is pd.DataFrame:
            df = dataset
        self.data = df.drop(targetName, axis=1)
        self.target = df[targetName]
        self.trainingData['x'], self.testData['x'],  \
        self.trainingData['y'], self.testData['y'] = \
            train_test_split(
                self.data, self.target,
                test_size=test_size,
                random_state=np.floor(time.time()).astype(int)%4294967296
            )
