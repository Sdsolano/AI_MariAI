# app/ml/models/clean_risk_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
from typing import Dict, Tuple, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanRiskPredictionModel:
    """
    Sistema de predicción de riesgo académico SIN data leakage
    
    Features permitidas: Solo datos académicos puros sin derivaciones del target
    Target: is_high_risk (basado en risk_level original)
    """
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.training_history = []
        self.feature_stats = {}
        
        # Features LIMPIAS - Sin data leakage
        self.allowed_features = [
            # Datos demográficos básicos
            'edad', 'estrato',
            
            # Métricas académicas puras
            'total_grades', 'average_grade', 'grade_std', 
            'min_grade', 'max_grade', 'subjects_count',
            
            # Conteos de notas por categoría (datos puros)
            'grades_excellent', 'grades_good', 'grades_satisfactory', 
            'grades_low', 'grades_very_low',
            
            # Métricas temporales
            'meses_activos', 'grades_per_month', 'days_since_last_activity',
            
            # Percentiles derivados (están OK porque son transformaciones lineales)
            'pct_excellent', 'pct_good', 'pct_satisfactory', 'pct_low_combined',
            
            # Métricas de variabilidad
            'grade_range', 'grade_consistency'
        ]
        
        # Features PROHIBIDAS - Contienen data leakage
        self.forbidden_features = [
            'risk_score',  # Calculado directamente del target
            'academic_performance_score',  # Derivado del target
            'risk_level',  # Es el target categórico
            'is_high_risk'  # Es el target binario
        ]
        
        # Configuraciones optimizadas de modelos
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
    
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """Cargar datos y eliminar data leakage"""
        logger.info(f"🔍 Loading and cleaning data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"✅ Raw data loaded: {df.shape[0]} students, {df.shape[1]} features")
        
        # Verificar features prohibidas
        forbidden_found = [col for col in df.columns if col in self.forbidden_features]
        if forbidden_found:
            logger.warning(f"⚠️  Found forbidden features: {forbidden_found}")
        
        # Mantener solo features permitidas + identificadores + target
        keep_cols = (self.allowed_features + 
                    ['student_id', 'nombre_completo', 'risk_level'])
        
        available_cols = [col for col in keep_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Crear target binario LIMPIO (sin usar risk_score)
        # Mapeo directo desde risk_level categórico
        df_clean['is_high_risk'] = (
            df_clean['risk_level'].isin(['ALTO', 'CRITICO'])
        ).astype(int)
        
        logger.info(f"🧹 Clean data: {df_clean.shape[0]} students, {df_clean.shape[1]} features")
        logger.info(f"   Features removed: {set(df.columns) - set(df_clean.columns)}")
        
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preparar features para entrenamiento"""
        logger.info("🔧 Preparing clean features...")
        
        # Solo features numéricas permitidas
        feature_cols = [col for col in self.allowed_features if col in df.columns]
        X = df[feature_cols].copy()
        
        # Target binario
        y = df['is_high_risk'].astype(int)
        
        # Verificar tipos de datos
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"⚠️  Converting to numeric: {non_numeric_cols}")
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        
        # Estadísticas de features
        self.feature_stats = {
            'means': X.mean().to_dict(),
            'stds': X.std().to_dict(),
            'mins': X.min().to_dict(),
            'maxs': X.max().to_dict()
        }
        
        logger.info(f"✅ Clean features prepared: {X.shape[1]} features")
        logger.info(f"   Feature columns: {list(X.columns)}")
        logger.info(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.3) -> Dict[str, Any]:
        """Entrenar modelos con features limpias"""
        logger.info("🚀 Starting CLEAN model training...")
        
        results = {}
        
        # Split con más datos para test (dataset pequeño)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training set: {X_train.shape[0]} samples")
        logger.info(f"📊 Test set: {X_test.shape[0]} samples")
        logger.info(f"📊 Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"📊 Test target distribution: {y_test.value_counts().to_dict()}")
        
        # Entrenar modelos
        for model_name, config in self.model_configs.items():
            logger.info(f"🤖 Training {model_name}...")
            
            # Crear pipeline
            if model_name == 'logistic_regression':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', config['model'])
                ])
                param_grid = {f'classifier__{k}': v for k, v in config['params'].items()}
            else:
                pipeline = Pipeline([
                    ('classifier', config['model'])
                ])
                param_grid = {f'classifier__{k}': v for k, v in config['params'].items()}
            
            # Grid search con validación cruzada más robusta
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=1, verbose=0, return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Mejor modelo
            best_model = grid_search.best_estimator_
            
            # Predicciones
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Métricas más robustas
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='roc_auc')
            test_auc = roc_auc_score(y_test, y_proba)
            
            # Verificar overfitting
            train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
            overfitting_gap = train_auc - test_auc
            
            # Guardar modelo
            self.models[model_name] = best_model
            
            # Feature importance
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                importance = best_model.named_steps['classifier'].feature_importances_
                self.feature_importance[model_name] = dict(zip(X.columns, importance))
            
            # Resultados
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'train_auc': train_auc,
                'test_auc': test_auc,
                'overfitting_gap': overfitting_gap,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"✅ {model_name}:")
            logger.info(f"   CV AUC: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            logger.info(f"   Test AUC: {test_auc:.3f}")
            logger.info(f"   Overfitting Gap: {overfitting_gap:.3f}")
        
        # Mejor modelo
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        logger.info(f"🏆 Best model: {best_model_name} (AUC: {results[best_model_name]['test_auc']:.3f})")
        
        # Información de entrenamiento
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'features_used': list(X.columns),
            'test_size': test_size,
            'best_model': best_model_name
        }
        self.training_history.append(training_info)
        
        return results
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 10) -> Dict[str, float]:
        """Obtener importancia de features limpias"""
        if model_name and model_name in self.feature_importance:
            importance = self.feature_importance[model_name]
        elif 'random_forest' in self.feature_importance:
            importance = self.feature_importance['random_forest']
        else:
            logger.warning("No feature importance available")
            return {}
        
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return dict(list(sorted_importance.items())[:top_n])
    
    def predict_risk(self, student_data: Dict[str, Any], model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predecir riesgo con features limpias"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        # Preparar datos de entrada con TODAS las features esperadas
        df = pd.DataFrame([student_data])
        
        # Crear DataFrame con todas las features esperadas por el modelo
        expected_features = [col for col in self.allowed_features if col in self.feature_stats['means']]
        X = pd.DataFrame(columns=expected_features, index=[0])
        
        # Llenar con datos proporcionados
        for col in expected_features:
            if col in student_data:
                X[col] = student_data[col]
            else:
                # Usar media de entrenamiento para features faltantes
                X[col] = self.feature_stats['means'].get(col, 0)
        
        # Asegurar tipos numéricos
        X = X.astype(float)
        
        # Predicción
        risk_binary = model.predict(X)[0]
        risk_proba = model.predict_proba(X)[0, 1]
        
        # Categorización basada en probabilidad
        if risk_proba < 0.2:
            risk_category = "BAJO"
        elif risk_proba < 0.4:
            risk_category = "MEDIO"
        elif risk_proba < 0.7:
            risk_category = "ALTO"
        else:
            risk_category = "CRITICO"
        
        return {
            'student_id': student_data.get('student_id', 'unknown'),
            'risk_binary': int(risk_binary),
            'risk_probability': float(risk_proba),
            'risk_category': risk_category,
            'confidence': float(max(risk_proba, 1-risk_proba)),
            'model_used': model_name,
            'features_used': len(expected_features)
        }
    
    def save_models(self, save_dir: str = "models/"):
        """Guardar modelos limpios"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"{save_dir}mari_ai_clean_{model_name}_{timestamp}.joblib"
            joblib.dump(model, filename)
            logger.info(f"💾 Clean model saved: {filename}")
        
        # Guardar metadatos limpios
        metadata = {
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'allowed_features': self.allowed_features,
            'forbidden_features': self.forbidden_features,
            'feature_stats': self.feature_stats
        }
        
        metadata_file = f"{save_dir}mari_ai_clean_metadata_{timestamp}.joblib"
        joblib.dump(metadata, metadata_file)
        logger.info(f"💾 Clean metadata saved: {metadata_file}")

def main():
    """Entrenar modelos SIN data leakage"""
    logger.info("🚀 MARI AI - CLEAN RISK PREDICTION MODEL")
    logger.info("🧹 NO DATA LEAKAGE VERSION")
    logger.info("=" * 60)
    
    # Inicializar modelo limpio
    clean_model = CleanRiskPredictionModel()
    
    # Cargar y limpiar datos
    data_file = "data/ml/mari_ai_ml_dataset_main.csv"
    df = clean_model.load_and_clean_data(data_file)
    
    # Preparar features limpias
    X, y = clean_model.prepare_features(df)
    
    # Entrenar modelos
    results = clean_model.train_models(X, y)
    
    # Mostrar resultados
    logger.info("\n📊 CLEAN TRAINING RESULTS:")
    logger.info("-" * 50)
    for model_name, metrics in results.items():
        logger.info(f"🤖 {model_name.upper()}:")
        logger.info(f"   CV AUC: {metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}")
        logger.info(f"   Train AUC: {metrics['train_auc']:.3f}")
        logger.info(f"   Test AUC: {metrics['test_auc']:.3f}")
        logger.info(f"   Overfitting: {metrics['overfitting_gap']:.3f}")
    
    # Feature importance
    logger.info("\n🎯 TOP FEATURES (Clean Random Forest):")
    logger.info("-" * 50)
    top_features = clean_model.get_feature_importance('random_forest', 10)
    for feature, importance in top_features.items():
        logger.info(f"   {feature}: {importance:.3f}")
    
    # Guardar modelos limpios
    clean_model.save_models()
    
    # Ejemplo de predicción limpia
    logger.info("\n🔮 CLEAN PREDICTION EXAMPLE:")
    logger.info("-" * 50)
    clean_example = {
        'student_id': 'EXAMPLE_001',
        'edad': 17,
        'estrato': 2.0,
        'total_grades': 45,
        'average_grade': 2.5,
        'grade_std': 1.2,
        'min_grade': 0.7,
        'max_grade': 4.8,
        'subjects_count': 8,
        'grades_excellent': 5,
        'grades_good': 8,
        'grades_satisfactory': 12,
        'grades_low': 15,
        'grades_very_low': 5,
        'meses_activos': 4,
        'grades_per_month': 11.25,
        'days_since_last_activity': 7,
        'pct_excellent': 0.111,
        'pct_good': 0.178,
        'pct_satisfactory': 0.267,
        'pct_low_combined': 0.444,
        'grade_range': 4.1,
        'grade_consistency': 0.8
    }
    
    prediction = clean_model.predict_risk(clean_example)
    logger.info(f"   Risk Category: {prediction['risk_category']}")
    logger.info(f"   Risk Probability: {prediction['risk_probability']:.3f}")
    logger.info(f"   Confidence: {prediction['confidence']:.3f}")
    logger.info(f"   Features Used: {prediction['features_used']}")
    
    logger.info("\n✅ CLEAN MODEL TRAINING COMPLETED!")

if __name__ == "__main__":
    main()