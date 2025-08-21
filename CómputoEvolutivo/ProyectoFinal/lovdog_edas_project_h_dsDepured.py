## Importation des bibliothèques
### Mathématiques
import pandas as pd # type: ignore
import numpy as np # type: ignore
import random
import os
### Visualisation
#import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
### Machine learning
#### Selection processus
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.model_selection import cross_val_score # type: ignore
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
        _ModelSelectionClassesNames_.append(name)
        _ModelSelectionClassesObjects_.append(self)
        self.name=name
        self.data   = None
        self.target = None
        self.gridParameters = {
            'SVC': {
                # Regularization
                'clf__C': [ i*0.001 for i in range(1, 100, 2) ],
                'clf__kernel': [ 'poly', 'sigmoid', 'rbf' ],
                'clf__gamma': [ 'scale', 'auto' ],
            },
            'KNeighbors': {
                'clf__n_neighbors': [ i for i in range(5,100,5) ],
                'clf__weights': [ 'uniform', 'distance' ],
            },
            'RandomForest': {
                'clf__min_samples_split': [ i for i in range(2,30,2) ],
                # 2, 4, 8 and 16 estimators for each feature
                'clf__n_estimators': [ i for i in range(15,100, 5) ],
                # To be fair, depth can cause unwanted complexity,
                ## The best depth should be between 3 and 27
                'clf__max_depth': [ i for i in range(3, 30, 2) ],
            },
            'LogisticRegression': {
                # Regularization
                'clf__C': [ i*0.001 for i in range(1, 100, 2) ],
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

        if csv is not None:
            self.loadDataset(csv)

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

    def saveSummary(self):
        pd.DataFrame(self.summary).to_csv(f"summary_{self.name}.csv", index=False)

    def getBestModel(self):
        return self.summary[0]["best_estimator"]

    def printBestModel(self):
        print(self.getBestModel())

    def saveModels(self, path):
        for estimator in self.summary:
            joblib.dump(estimator["best_estimator"], f"{path}/{estimator['tag']}.joblib")

class FeatureSelectionEDA:
    def __init__(self, model_selection_instance=None, models=None, dataset=None, 
                 eID="default_exp", nGenerations=50, pSize=100, 
                 ssSizeRange=(5, 50), eliteFraction=0.2, mRate=0.1, 
                 classifierName="ensemble_eda", targetName="target", test_size=0.2):
        
        # Initialize trainingData and testData FIRST
        self.trainingData = {'x': None, 'y': None}
        self.testData = {'x': None, 'y': None}
        
        # Then the rest of your initialization
        self.EDAS_Config = {
            "experiment_id": eID,
            "num_generations": nGenerations,
            "population_size": pSize,
            "subset_size_range": ssSizeRange,
            "elite_fraction": eliteFraction,
            "mutation_rate": mRate,
            "random_state": np.floor(time.time()).astype(int)%4294967296,
            "classifier_name": classifierName,
            "test_size": test_size
        }
        
        self.model_paths = []
        self.models = []
        self.data = None
        self.target = None
        self.cache = {}
        self.results = {
            'generation': [], 'best_fitness': [], 'avg_fitness': [],
            'worst_fitness': [], 'best_individual': [], 
            'best_feature_count': [], 'best_features': []
        }
        
        # Handle initialization
        if model_selection_instance is not None:
            self._init_from_model_selection(model_selection_instance)
        elif models is not None and dataset is not None:
            self.loadDataset(dataset, targetName, test_size)  # This should populate trainingData
            self.loadModel(models)
        else:
            raise ValueError("Must provide either model_selection_instance or both models and dataset")
        
        self._setup_deap()

    def _init_from_model_selection(self, model_selection):
        """Initialize from a ModelSelection instance"""
        # Use the same data as the model selection
        self.data = model_selection.data
        self.target = model_selection.target
        
        # Create train/test split using the same data
        (self.trainingData['x'], self.testData['x'], 
         self.trainingData['y'], self.testData['y']) = train_test_split(
            self.data, self.target,
            test_size=self.EDAS_Config["test_size"],
            random_state=self.EDAS_Config["random_state"]
        )
        
        # Load all models from the model selection summary
        for model_info in model_selection.summary:
            self.models.append(model_info["best_estimator"])
        
        print(f"Initialized from ModelSelection '{model_selection.name}'")
        print(f"Loaded {len(self.models)} optimized models")
        print(f"Training data shape: {self.trainingData['x'].shape}")
        print(f"Test data shape: {self.testData['x'].shape}")

    def loadModel(self, path=None, append=False):
        """Load model(s) from path(s)"""
        if path is None:
            return
            
        if append:
            if isinstance(path, list):
                self.model_paths.extend(path)
            else:
                self.model_paths.append(path)
        else:
            self.model_paths = path if isinstance(path, list) else [path]
            self.models = []  # Reset models when not appending

        # Clear models if not appending
        if not append:
            self.models = []

        # Handle directory input
        if isinstance(path, str) and os.path.isdir(path):
            model_files = [f for f in os.listdir(path) if f.endswith(('.joblib', '.pkl', '.sav'))]
            for file in model_files:
                try:
                    self.models.append(joblib.load(os.path.join(path, file)))
                except Exception as e:
                    print(f"Error loading model {file}: {e}")

        # Handle list of directories
        elif isinstance(path, list) and all(os.path.isdir(p) for p in path):
            for directory in path:
                model_files = [f for f in os.listdir(directory) if f.endswith(('.joblib', '.pkl', '.sav'))]
                for file in model_files:
                    try:
                        self.models.append(joblib.load(os.path.join(directory, file)))
                    except Exception as e:
                        print(f"Error loading model {file} from {directory}: {e}")

        # Handle list of model files
        elif isinstance(path, list) and all(not os.path.isdir(p) for p in path):
            for model_file in path:
                try:
                    self.models.append(joblib.load(model_file))
                except Exception as e:
                    print(f"Error loading model {model_file}: {e}")

        # Handle single model file
        elif isinstance(path, str) and not os.path.isdir(path):
            try:
                self.models.append(joblib.load(path))
            except Exception as e:
                print(f"Error loading model {path}: {e}")

        print(f"Loaded {len(self.models)} models")

    def loadDataset(self, dataset, targetName="target", test_size=0.2):
        """Load and split dataset"""
        if isinstance(dataset, str):
            df = pd.read_csv(dataset)
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise ValueError("Dataset must be a file path or DataFrame")
        
        self.data = df.drop(targetName, axis=1)
        self.target = df[targetName]
        
        # Create train/test split
        (self.trainingData['x'], self.testData['x'], 
         self.trainingData['y'], self.testData['y']) = train_test_split(
            self.data, self.target,
            test_size=test_size,
            random_state=self.EDAS_Config["random_state"]
        )
        
        print(f"Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        print(f"Training set: {self.trainingData['x'].shape[0]} samples")
        print(f"Test set: {self.testData['x'].shape[0]} samples")

    def _setup_deap(self):
        """Setup DEAP framework components"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Register attribute and individual generators
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_bool, 
                             n=self.data.shape[1])
        
        # Register population generator
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        # Register selection operator (using elite fraction)
        elite_size = int(self.EDAS_Config["population_size"] * 
                        self.EDAS_Config["elite_fraction"])
        self.toolbox.register("select", tools.selBest, k=elite_size)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual's fitness using all loaded models.
        Use training data for cross-validation to avoid data leakage.
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
        
        # Apply feature selection to TRAINING data
        selected_features = self.trainingData['x'].columns[mask]
        X_subset = self.trainingData['x'][selected_features]
        
        # Evaluate using all models and aggregate scores
        scores = []
        for model in self.models:
            try:
                # Use training data for cross-validation
                score = cross_val_score(model, X_subset, self.trainingData['y'], 
                                      cv=3, scoring='accuracy').mean()
                scores.append(score)
            except Exception as e:
                print(f"Error evaluating model: {e}")
                scores.append(0.0)
        
        # Use average score across all models as fitness
        fitness = (np.mean(scores),)
        self.cache[ind_tuple] = fitness
        return fitness

    # Add your EDAS algorithm methods here next
    def _edas_variation(self, population):
        """Implement your EDAS variation logic here"""
        # This is where you'll implement the EDAS-specific logic
        # for creating new individuals based on the elite population
        pass

    def run(self):
        """Main method to run the EDAS algorithm with result tracking"""
        # Initialize population
        population = self.toolbox.population(n=self.EDAS_Config["population_size"])
        
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Main evolutionary loop
        for gen in range(self.EDAS_Config["num_generations"]):
            # Your EDAS logic goes here
            # 1. Select elite individuals
            # 2. Create new population using EDAS variation
            # 3. Evaluate new individuals
            # 4. Replace population
            
            # Track statistics for this generation
            fits = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            
            # Store results
            self._record_generation_results(gen, fits, best_ind)
            
            print(f"Generation {gen}: Best={max(fits):.4f}, Avg={np.mean(fits):.4f}")
        
        self.final_population = population
        self.best_individual = tools.selBest(population, 1)[0]
        
        return population

    def _record_generation_results(self, generation, fitnesses, best_individual):
        """Record results for a single generation"""
        mask = np.array(best_individual, dtype=bool)
        selected_features = self.data.columns[mask].tolist()
        
        self.results['generation'].append(generation)
        self.results['best_fitness'].append(max(fitnesses))
        self.results['avg_fitness'].append(np.mean(fitnesses))
        self.results['worst_fitness'].append(min(fitnesses))
        self.results['best_individual'].append(tuple(best_individual))
        self.results['best_feature_count'].append(np.sum(best_individual))
        self.results['best_features'].append(selected_features)

    def get_results_dataframe(self):
        """Return results as a pandas DataFrame"""
        return pd.DataFrame(self.results)

    def get_final_results(self):
        """Get comprehensive final results"""
        if self.best_individual is None:
            return None
        
        best_mask = np.array(self.best_individual, dtype=bool)
        selected_features = self.data.columns[best_mask].tolist()
        best_fitness = self.best_individual.fitness.values[0]
        
        return {
            'best_fitness': best_fitness,
            'selected_feature_count': len(selected_features),
            'selected_features': selected_features,
            'feature_mask': best_mask,
            'total_generations': self.EDAS_Config["num_generations"],
            'population_size': self.EDAS_Config["population_size"],
            'models_used': [type(model).__name__ for model in self.models]
        }

    def export_to_csv(self, filename="eda_results.csv"):
        """Export results to CSV file"""
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    def export_to_excel(self, filename="eda_results.xlsx"):
        """Export results to Excel file"""
        df = self.get_results_dataframe()
        df.to_excel(filename, index=False)
        print(f"Results exported to {filename}")

    def export_final_report(self, filename="final_report.txt"):
        """Export a comprehensive final report"""
        final_results = self.get_final_results()
        if final_results is None:
            print("No results to export. Run the algorithm first.")
            return
        
        with open(filename, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("FEATURE SELECTION EDA - FINAL REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment ID: {self.EDAS_Config['experiment_id']}\n")
            f.write(f"Best Fitness Score: {final_results['best_fitness']:.4f}\n")
            f.write(f"Number of Selected Features: {final_results['selected_feature_count']}\n")
            f.write(f"Total Generations: {final_results['total_generations']}\n")
            f.write(f"Population Size: {final_results['population_size']}\n")
            f.write(f"Models Used: {', '.join(final_results['models_used'])}\n\n")
            
            f.write("SELECTED FEATURES:\n")
            f.write("-" * 30 + "\n")
            for i, feature in enumerate(final_results['selected_features'], 1):
                f.write(f"{i:2d}. {feature}\n")
        
        print(f"Final report exported to {filename}")

    def get_best_feature_set(self):
        """Get the best feature set found"""
        if self.best_individual is None:
            return None
        mask = np.array(self.best_individual, dtype=bool)
        return self.data.columns[mask].tolist()

    def get_feature_importance(self):
        """Calculate how often each feature appears in best individuals"""
        if not self.results['best_individual']:
            return None
        
        feature_counts = np.zeros(self.n_features)
        for individual in self.results['best_individual']:
            feature_counts += np.array(individual)
        
        importance_df = pd.DataFrame({
            'feature': self.data.columns,
            'selection_frequency': feature_counts / len(self.results['best_individual']),
            'final_selection': np.array(self.best_individual, dtype=bool)
        })
        return importance_df.sort_values('selection_frequency', ascending=False)

    def compare_with_original_features(self, model_selection_instance):
        """Compare EDA-selected features with original model performance"""
        original_performance = {}
        eda_performance = {}
        
        # Get original performance from model selection
        for model_info in model_selection_instance.summary:
            tag = model_info["tag"]
            original_performance[tag] = model_info["best_accuracy"]
        
        # Get performance with EDA-selected features
        best_mask = np.array(self.best_individual, dtype=bool)
        selected_features = self.data.columns[best_mask]
        X_eda = self.data[selected_features]
        
        for model in self.models:
            model_name = type(model.named_steps['clf']).__name__
            cv_score = cross_val_score(model, X_eda, self.target, cv=5, scoring='accuracy').mean()
            eda_performance[model_name] = cv_score
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name in original_performance:
            if model_name in eda_performance:
                improvement = eda_performance[model_name] - original_performance[model_name]
                comparison_data.append({
                    'Model': model_name,
                    'Original_Accuracy': original_performance[model_name],
                    'EDA_Accuracy': eda_performance[model_name],
                    'Improvement': improvement,
                    'Improvement_Percent': (improvement / original_performance[model_name]) * 100
                })
        
        return pd.DataFrame(comparison_data)

    def evaluate_on_test(self, individual=None):
        """Evaluate the best individual on the test set"""
        if individual is None:
            if self.best_individual is None:
                raise ValueError("No best individual found. Run the algorithm first.")
            individual = self.best_individual
        
        mask = np.array(individual, dtype=bool)
        selected_features = self.testData['x'].columns[mask]
        X_test_subset = self.testData['x'][selected_features]
        
        test_scores = []
        for model in self.models:
            try:
                # Train on full training set and test on test set
                X_train_subset = self.trainingData['x'][selected_features]
                model.fit(X_train_subset, self.trainingData['y'])
                score = model.score(X_test_subset, self.testData['y'])
                test_scores.append(score)
            except Exception as e:
                print(f"Error testing model: {e}")
                test_scores.append(0.0)
        
        return np.mean(test_scores)
