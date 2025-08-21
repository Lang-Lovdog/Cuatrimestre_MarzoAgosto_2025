## Importation des bibliothèques
### Mathématiques
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
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
from sklearn.metrics import cross_val_score # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import f1_score # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay # type: ignore
### Temps (pour random_state)
import time # type: ignore
## Model import/export
import joblib # type: ignore
# DEAP for EDAs
from deap import base, creator, tools  # type: ignore



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
        self.model_paths = models if type(models) is list else [models]
        self.models = []
        self.EDAS_Config = {
            "experiment_id": eID,
            "num_generations": nGenerations,
            "population_size": pSize,
            "subset_size_range": ssSizeRange,
            "elite_fraction": eliteFraction,

            "random_state": randomState,
            "classifier_name": classifierName
        }
        self.data   : pd.DataFrame = None
        self.target : pd.DataFrame = None

    def _setup_deap(self):
        """
        Sets up DEAP creator classes and toolbox with registered operators.
        """
        # Create Fitness and Individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator: 0 or 1 with probability
        # We can use random.randint for binary attributes
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Individual generator: create individual of length n_features
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_bool, 
                             n=self.n_features)
        
        # Population generator: create list of individuals
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # Register the evaluation function with our data and models
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        # Register selection method (elite selection based on elite_fraction)
        # tools.selBest will select the top n individuals
        elite_size = int(self.EDAS_Config["population_size"] * 
                        self.EDAS_Config["elite_fraction"])
        self.toolbox.register("select", tools.selBest, k=elite_size)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual's fitness using all loaded models.
        """
        # Convert to tuple for caching
        ind_tuple = tuple(individual)
        if ind_tuple in self.cache:
            return self.cache[ind_tuple]
        
        # Convert individual to boolean mask
        mask = np.array(individual, dtype=bool)
        
        # Check if any features are selected
        if not np.any(mask):
            fitness = (0.0,)
            self.cache[ind_tuple] = fitness
            return fitness
        
        # Apply feature selection
        selected_features = self.data.columns[mask]
        X_subset = self.data[selected_features]
        
        # Evaluate using all models and aggregate scores
        scores = []
        for model in self.models:
            try:
                score = cross_val_score(model, X_subset, self.target, 
                                      cv=3, scoring='accuracy').mean()
                scores.append(score)
            except Exception as e:
                print(f"Error evaluating model: {e}")
                scores.append(0.0)
        
        # Use average score across all models as fitness
        fitness = (np.mean(scores),)
        self.cache[ind_tuple] = fitness
        return fitness

    def eda_evalfunc(self, model, individual):
        """
        Evaluates a feature subset individual.
        
        Args:
            individual (list): A binary list of length n_features. 1 means select the feature.
            X (pd.DataFrame): The complete DataFrame of features.
            y (pd.Series): The complete target Series.
            model: The pre-trained scikit-learn model.
            
        Returns:
            tuple: A tuple containing the fitness score (e.g., accuracy).
        """
        
        # 1. Convert the individual to a boolean mask
        #    individual is a list of 0s and 1s from DEAP
        mask = np.array(individual, dtype=bool)
        
        # 2. Check if any feature is selected. Avoid errors with zero features.
        if not np.any(mask):
            return 0.0, # Return a terrible fitness if no features are selected
        
        # 3. Apply the mask to get the selected feature names
        #    This gets the column names where the mask is True
        selected_features = self.data.columns[mask]
        
        # 4. Create the subset DataFrame with only the selected features
        X_subset = self.data[selected_features]
        
        # 5. Evaluate the model using cross-validation on the ENTIRE dataset (X_subset, y)
        #    This trains and validates on all data, which is correct for feature selection
        #    with a fixed model.
        score = cross_val_score(model, X_subset, self.target, cv=8, scoring='accuracy').mean()
        
        return score

    def loadModel(self, path=None, append=False):
        if path is not None:
            if append:
                self.model_paths=self.model_paths.append(path)
            else:
                self.model_paths= path if type(path) is list else [path]

            if os.path.isdir(path) and type(path) is str:
                self.models=[
                    joblib.load(f"{path}/{file}")
                    for file in os.listdir(path)
                ]
            if os.path.isdir(path) and type(path) is list:
                for dir in path:
                    self.models=[
                        joblib.load(f"{dir}/{file}")
                        for file in os.listdir(dir)
                    ]
            elif type(path) is list and not os.path.isdir(path[0]):
                self.models=[
                    joblib.load(file)
                    for file in path
                ]
            elif not os.path.isdir(path) and type(path) is str:
                self.models=[
                    joblib.load(path)
                ]

    def loadDataset(self, dataset, targetName="target", test_size=0.2):
        if type(dataset) is str:
            df = pd.read_csv(dataset)
        elif type(dataset) is pd.DataFrame:
            df = dataset
        self.data = df.drop(targetName, axis=1)
        self.target = df[targetName]
