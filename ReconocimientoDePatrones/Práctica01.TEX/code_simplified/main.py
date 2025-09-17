import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import os
from datetime import datetime

# Importar las clases necesarias
from LovdogDF import LovdogDataFrames
from LovdogGLCM import GLCMFeatures
from LovdogGLRL import GLRFeatures
from LovdogSDH import SDHFeaturesMultiAngle
from LovdogMS import ModelSelection

# Create results directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"experiment_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

def get_safe_grid_params(n_samples):
    """
    Obtener parámetros de grid seguros basados en el número de muestras
    """
    # Ajustar n_neighbors para KNeighborsClassifier
    max_neighbors = min(20, n_samples // 2)  # Máximo 20 o la mitad de las muestras
    
    safe_params = {
        'SVC': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto'],
        },
        'KNeighbors': {
            'clf__n_neighbors': [3, 5, 7, max(3, min(9, max_neighbors))],
            'clf__weights': ['uniform', 'distance'],
        },
        'RandomForest': {
            'clf__min_samples_split': [2, 4, 8],
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [None, 5, 10],
        },
        'LogisticRegression': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear'],
        },
    }
    
    return safe_params

def save_dataset_info(data_dict, filename):
    """Save dataset information to CSV"""
    info_data = []
    for class_name, frames in data_dict.items():
        info_data.append({
            'Class': class_name,
            'Samples': len(frames),
            'Image_Shape': frames[0].shape if len(frames) > 0 else 'N/A'
        })
    
    df = pd.DataFrame(info_data)
    df.to_csv(filename, index=False)
    return df

def save_features(X, y, feature_names, filename):
    """Save feature matrix to CSV with labels"""
    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = y
    df.to_csv(filename, index=False)
    return df

def save_confusion_matrix_png(confusion_matrix, class_names, filename):
    """Save confusion matrix as PNG image"""
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_learning_curve_png(model, X, y, filename):
    """Save learning curve as PNG image"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score', linewidth=2)
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score', linewidth=2)
    plt.title('Learning Curve', fontsize=14)
    plt.xlabel('Training examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_model_results(model_selection, case_name, feature_extractor=None):
    """Save all model results for a case"""
    case_dir = os.path.join(results_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = model_selection.getMetricsDataFrame()
    metrics_df.to_csv(os.path.join(case_dir, "metrics.csv"), index=False)
    
    # Save best model parameters
    best_params = model_selection.results[model_selection.best_model_tag]['best_params']
    with open(os.path.join(case_dir, "best_params.txt"), 'w') as f:
        f.write(f"Best Model: {model_selection.best_model_tag}\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Test Accuracy: {model_selection.results[model_selection.best_model_tag]['test_accuracy']:.4f}\n")
    
    # Save confusion matrix as CSV and PNG
    cm = model_selection.results[model_selection.best_model_tag]['confusion_matrix']
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(case_dir, "confusion_matrix.csv"), index=False)
    
    # Save confusion matrix as PNG
    class_names = model_selection.class_names if hasattr(model_selection, 'class_names') else [f"Class {i}" for i in range(cm.shape[0])]
    save_confusion_matrix_png(cm, class_names, os.path.join(case_dir, "confusion_matrix.png"))
    
    # Save learning curve as PNG
    best_model = model_selection.results[model_selection.best_model_tag]['best_estimator']
    save_learning_curve_png(best_model, model_selection.data, model_selection.target, 
                           os.path.join(case_dir, "learning_curve.png"))
    
    # Save feature names if available
    if feature_extractor:
        feature_names = feature_extractor.get_feature_names()
        with open(os.path.join(case_dir, "feature_names.txt"), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
    
    return case_dir

def main():
    try:
        # Cargar los datos
        print("Cargando datos...")
        data_porter = LovdogDataFrames('images.json')
        data_dict = data_porter.get_data_dict()
        
        # Save dataset information
        dataset_info_file = os.path.join(results_dir, "dataset_info.csv")
        dataset_info = save_dataset_info(data_dict, dataset_info_file)
        print(f"Dataset information saved to {dataset_info_file}")
        
        # Verificar que los datos se cargaron correctamente
        if not data_dict:
            print("ERROR: No se pudieron cargar los datos. Verifica el archivo images.json y las rutas de las imágenes.")
            return
            
        print(f"Datos cargados: {len(data_dict)} clases")
        total_samples = 0
        for class_name, frames in data_dict.items():
            print(f"  Clase '{class_name}': {len(frames)} imágenes")
            total_samples += len(frames)
        
        # Obtener parámetros seguros basados en el tamaño del dataset
        safe_params = get_safe_grid_params(total_samples)
        print(f"Usando parámetros seguros para dataset pequeño (n={total_samples})")
        
        # Caso 1: GLCM sin y con PCA
        print("\n=== CASO 1: GLCM ===")
        try:
            glcm_extractor = GLCMFeatures()
            X_glcm, y_glcm = glcm_extractor.fit_transform(data_dict)
            print(f"Características GLCM extraídas: {X_glcm.shape}")
            
            # Save GLCM features
            glcm_features_file = os.path.join(results_dir, "glcm_features.csv")
            save_features(X_glcm, y_glcm, glcm_extractor.get_feature_names(), glcm_features_file)
            print(f"GLCM features saved to {glcm_features_file}")
            
            # Modelos sin PCA
            ms_glcm = ModelSelection("GLCM_sin_PCA", ml_data_tuple=(X_glcm, y_glcm), 
                                    grid_parameters_dict=safe_params)
            ms_glcm.selectModels(use_pca=False)
            
            # Save results for GLCM without PCA
            glcm_no_pca_dir = save_model_results(ms_glcm, "case1_glcm_no_pca", glcm_extractor)
            print(f"GLCM without PCA results saved to {glcm_no_pca_dir}")
            
            # Modelos con PCA
            ms_glcm_pca = ModelSelection("GLCM_con_PCA", ml_data_tuple=(X_glcm, y_glcm),
                                        grid_parameters_dict=safe_params)
            ms_glcm_pca.selectModels(use_pca=True, n_components=min(0.95, X_glcm.shape[1]))
            
            # Save results for GLCM with PCA
            glcm_pca_dir = save_model_results(ms_glcm_pca, "case1_glcm_with_pca", glcm_extractor)
            print(f"GLCM with PCA results saved to {glcm_pca_dir}")
            
            # Determinar el mejor caso y mostrar su matriz de confusión
            glcm_no_pca_acc = ms_glcm.results[ms_glcm.best_model_tag]['test_accuracy']
            glcm_pca_acc = ms_glcm_pca.results[ms_glcm_pca.best_model_tag]['test_accuracy']
            
            if glcm_no_pca_acc >= glcm_pca_acc:
                best_glcm = ms_glcm
                print(f"Mejor configuración GLCM: Sin PCA (Accuracy: {glcm_no_pca_acc:.4f})")
            else:
                best_glcm = ms_glcm_pca
                print(f"Mejor configuración GLCM: Con PCA (Accuracy: {glcm_pca_acc:.4f})")
                
        except Exception as e:
            print(f"Error en GLCM: {e}")
            traceback.print_exc()
            
        # Caso 2: GLR sin y con PCA
        print("\n=== CASO 2: GLR ===")
        try:
            glr_extractor = GLRFeatures()
            X_glr, y_glr = glr_extractor.fit_transform(data_dict)
            print(f"Características GLR extraídas: {X_glr.shape}")
            
            # Save GLR features
            glr_features_file = os.path.join(results_dir, "glr_features.csv")
            save_features(X_glr, y_glr, glr_extractor.get_feature_names(), glr_features_file)
            print(f"GLR features saved to {glr_features_file}")
            
            # Modelos sin PCA
            ms_glr = ModelSelection("GLR_sin_PCA", ml_data_tuple=(X_glr, y_glr),
                                   grid_parameters_dict=safe_params)
            ms_glr.selectModels(use_pca=False)
            
            # Save results for GLR without PCA
            glr_no_pca_dir = save_model_results(ms_glr, "case2_glr_no_pca", glr_extractor)
            print(f"GLR without PCA results saved to {glr_no_pca_dir}")
            
            # Modelos con PCA
            ms_glr_pca = ModelSelection("GLR_con_PCA", ml_data_tuple=(X_glr, y_glr),
                                       grid_parameters_dict=safe_params)
            ms_glr_pca.selectModels(use_pca=True, n_components=min(0.95, X_glr.shape[1]))
            
            # Save results for GLR with PCA
            glr_pca_dir = save_model_results(ms_glr_pca, "case2_glr_with_pca", glr_extractor)
            print(f"GLR with PCA results saved to {glr_pca_dir}")
            
            # Determinar el mejor caso
            glr_no_pca_acc = ms_glr.results[ms_glr.best_model_tag]['test_accuracy']
            glr_pca_acc = ms_glr_pca.results[ms_glr_pca.best_model_tag]['test_accuracy']
            
            if glr_no_pca_acc >= glr_pca_acc:
                best_glr = ms_glr
                print(f"Mejor configuración GLR: Sin PCA (Accuracy: {glr_no_pca_acc:.4f})")
            else:
                best_glr = ms_glr_pca
                print(f"Mejor configuración GLR: Con PCA (Accuracy: {glr_pca_acc:.4f})")
                
        except Exception as e:
            print(f"Error en GLR: {e}")
            traceback.print_exc()
            
        # Caso 3: SDH sin y con PCA
        print("\n=== CASO 3: SDH ===")
        try:
            sdh_extractor = SDHFeaturesMultiAngle()
            X_sdh, y_sdh = sdh_extractor.fit_transform(data_dict)
            print(f"Características SDH extraídas: {X_sdh.shape}")
            
            # Save SDH features
            sdh_features_file = os.path.join(results_dir, "sdh_features.csv")
            save_features(X_sdh, y_sdh, sdh_extractor.get_feature_names(), sdh_features_file)
            print(f"SDH features saved to {sdh_features_file}")
            
            # Modelos sin PCA
            ms_sdh = ModelSelection("SDH_sin_PCA", ml_data_tuple=(X_sdh, y_sdh),
                                   grid_parameters_dict=safe_params)
            ms_sdh.selectModels(use_pca=False)
            
            # Save results for SDH without PCA
            sdh_no_pca_dir = save_model_results(ms_sdh, "case3_sdh_no_pca", sdh_extractor)
            print(f"SDH without PCA results saved to {sdh_no_pca_dir}")
            
            # Modelos con PCA
            ms_sdh_pca = ModelSelection("SDH_con_PCA", ml_data_tuple=(X_sdh, y_sdh),
                                       grid_parameters_dict=safe_params)
            ms_sdh_pca.selectModels(use_pca=True, n_components=min(0.95, X_sdh.shape[1]))
            
            # Save results for SDH with PCA
            sdh_pca_dir = save_model_results(ms_sdh_pca, "case3_sdh_with_pca", sdh_extractor)
            print(f"SDH with PCA results saved to {sdh_pca_dir}")
            
            # Determinar el mejor caso
            sdh_no_pca_acc = ms_sdh.results[ms_sdh.best_model_tag]['test_accuracy']
            sdh_pca_acc = ms_sdh_pca.results[ms_sdh_pca.best_model_tag]['test_accuracy']
            
            if sdh_no_pca_acc >= sdh_pca_acc:
                best_sdh = ms_sdh
                print(f"Mejor configuración SDH: Sin PCA (Accuracy: {sdh_no_pca_acc:.4f})")
            else:
                best_sdh = ms_sdh_pca
                print(f"Mejor configuración SDH: Con PCA (Accuracy: {sdh_pca_acc:.4f})")
                
        except Exception as e:
            print(f"Error en SDH: {e}")
            traceback.print_exc()
            
        # Solo continuar si al menos un extractor de características funcionó
        extractors_worked = any(['X_glcm' in locals(), 'X_glr' in locals(), 'X_sdh' in locals()])
        
        if not extractors_worked:
            print("ERROR: Ningún extractor de características funcionó. Verifica tus datos e implementación.")
            return
            
        # Caso 4: Combinar todas las características con PCA y LDA
        print("\n=== CASO 4: Características Combinadas ===")
        try:
            # Combinar características disponibles
            available_features = []
            feature_names = []
            
            if 'X_glcm' in locals():
                available_features.append(X_glcm)
                feature_names.extend(glcm_extractor.get_feature_names())
            if 'X_glr' in locals():
                available_features.append(X_glr)
                feature_names.extend(glr_extractor.get_feature_names())
            if 'X_sdh' in locals():
                available_features.append(X_sdh)
                feature_names.extend(sdh_extractor.get_feature_names())
                
            X_combined = np.hstack(available_features)
            y_combined = y_glcm if 'y_glcm' in locals() else y_glr if 'y_glr' in locals() else y_sdh
            
            # Save combined features
            combined_features_file = os.path.join(results_dir, "combined_features.csv")
            save_features(X_combined, y_combined, feature_names, combined_features_file)
            print(f"Combined features saved to {combined_features_file}")
            
            # Ajustar n_components para PCA y LDA
            n_components_pca = min(0.95, X_combined.shape[1])
            n_components_lda = min(len(data_dict) - 1, X_combined.shape[1])
            
            # Modelos con PCA
            ms_combined_pca = ModelSelection("Combinado_PCA", ml_data_tuple=(X_combined, y_combined),
                                            grid_parameters_dict=safe_params)
            ms_combined_pca.selectModels(use_pca=True, n_components=n_components_pca)
            
            # Save results for combined features with PCA
            combined_pca_dir = save_model_results(ms_combined_pca, "case4_combined_with_pca")
            print(f"Combined features with PCA results saved to {combined_pca_dir}")
            
            # Modelos con LDA
            ms_combined_lda = ModelSelection("Combinado_LDA", ml_data_tuple=(X_combined, y_combined),
                                            grid_parameters_dict=safe_params)
            ms_combined_lda.selectModels(use_lda=True, n_components=n_components_lda)
            
            # Save results for combined features with LDA
            combined_lda_dir = save_model_results(ms_combined_lda, "case4_combined_with_lda")
            print(f"Combined features with LDA results saved to {combined_lda_dir}")
            
            # Determinar el mejor caso
            combined_pca_acc = ms_combined_pca.results[ms_combined_pca.best_model_tag]['test_accuracy']
            combined_lda_acc = ms_combined_lda.results[ms_combined_lda.best_model_tag]['test_accuracy']
            
            if combined_pca_acc >= combined_lda_acc:
                best_combined = ms_combined_pca
                print(f"Mejor configuración Combinada: PCA (Accuracy: {combined_pca_acc:.4f})")
            else:
                best_combined = ms_combined_lda
                print(f"Mejor configuración Combinada: LDA (Accuracy: {combined_lda_acc:.4f})")
                
        except Exception as e:
            print(f"Error en características combinadas: {e}")
            traceback.print_exc()
            
        # Caso 5: Mejor clasificador de casos 1-3 con LDA
        print("\n=== CASO 5: Mejor Clasificador con LDA ===")
        try:
            # Determinar el mejor de los casos 1-3
            best_accuracies = {}
            best_models = {}
            
            if 'best_glcm' in locals():
                best_accuracies['GLCM'] = best_glcm.results[best_glcm.best_model_tag]['test_accuracy']
                best_models['GLCM'] = best_glcm.best_model_tag
                
            if 'best_glr' in locals():
                best_accuracies['GLR'] = best_glr.results[best_glr.best_model_tag]['test_accuracy']
                best_models['GLR'] = best_glr.best_model_tag
                
            if 'best_sdh' in locals():
                best_accuracies['SDH'] = best_sdh.results[best_sdh.best_model_tag]['test_accuracy']
                best_models['SDH'] = best_sdh.best_model_tag
                
            if not best_accuracies:
                print("No hay modelos disponibles para comparar")
                return
                
            best_case_name = max(best_accuracies, key=best_accuracies.get)
            print(f"Mejor caso: {best_case_name} (Accuracy: {best_accuracies[best_case_name]:.4f})")
            
            # Seleccionar las características del mejor caso
            if best_case_name == 'GLCM':
                X_best = X_glcm
                y_best = y_glcm
                best_model_name = best_models['GLCM']
                classifier_obj = best_glcm.classifiers[best_model_name]
                grid_params = safe_params[best_model_name]  # Usar parámetros seguros
                feature_extractor = glcm_extractor
            elif best_case_name == 'GLR':
                X_best = X_glr
                y_best = y_glr
                best_model_name = best_models['GLR']
                classifier_obj = best_glr.classifiers[best_model_name]
                grid_params = safe_params[best_model_name]  # Usar parámetros seguros
                feature_extractor = glr_extractor
            else:  # SDH
                X_best = X_sdh
                y_best = y_sdh
                best_model_name = best_models['SDH']
                classifier_obj = best_sdh.classifiers[best_model_name]
                grid_params = safe_params[best_model_name]  # Usar parámetros seguros
                feature_extractor = sdh_extractor
            
            # Ajustar n_components para LDA
            n_components_lda = min(len(data_dict) - 1, X_best.shape[1])
            
            # Aplicar LDA al mejor caso
            ms_best_lda = ModelSelection(
                f"{best_case_name}_LDA", 
                ml_data_tuple=(X_best, y_best),
                classifiers_dict={best_model_name: classifier_obj},
                grid_parameters_dict={best_model_name: grid_params}
            )
            ms_best_lda.selectModels(use_lda=True, n_components=n_components_lda)
            
            # Save results for best case with LDA
            best_lda_dir = save_model_results(ms_best_lda, "case5_best_with_lda", feature_extractor)
            print(f"Best case with LDA results saved to {best_lda_dir}")
            
        except Exception as e:
            print(f"Error en el mejor clasificador con LDA: {e}")
            traceback.print_exc()
            
        # Create a summary report
        print("\n=== CREANDO REPORTE FINAL ===")
        create_summary_report(results_dir)
            
        print(f"\n=== ANÁLISIS COMPLETADO ===")
        print(f"Todos los resultados se han guardado en: {results_dir}")
        
    except Exception as e:
        print(f"Error general: {e}")
        traceback.print_exc()

def create_summary_report(results_dir):
    """Create a summary report of all experiments"""
    summary_data = []
    
    # Check each case and add to summary
    cases = [
        ("case1_glcm_no_pca", "GLCM sin PCA"),
        ("case1_glcm_with_pca", "GLCM con PCA"),
        ("case2_glr_no_pca", "GLR sin PCA"),
        ("case2_glr_with_pca", "GLR con PCA"),
        ("case3_sdh_no_pca", "SDH sin PCA"),
        ("case3_sdh_with_pca", "SDH con PCA"),
        ("case4_combined_with_pca", "Combinado con PCA"),
        ("case4_combined_with_lda", "Combinado con LDA"),
        ("case5_best_with_lda", "Mejor caso con LDA")
    ]
    
    for case_dir, case_name in cases:
        metrics_file = os.path.join(results_dir, case_dir, "metrics.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            best_row = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
            
            summary_data.append({
                'Caso': case_name,
                'Mejor Clasificador': best_row['Classifier'],
                'Precisión': best_row['Accuracy'],
                'F1-Score': best_row['F1_Score'],
                'AUC': best_row['AUC_Score'] if best_row['AUC_Score'] != 'N/A' else 'N/A'
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, "resumen_experimentos.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Resumen de experimentos guardado en: {summary_file}")
    return summary_df

if __name__ == "__main__":
    main()
