# scripts/validate_temporal_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalModelValidator:
    """
    Validador temporal para el modelo de riesgo académico
    
    Simula condiciones de producción donde predecimos el futuro
    basado en datos históricos
    """
    
    def __init__(self):
        self.results = {}
        self.feature_stability = {}
    
    def simulate_temporal_split(self, df: pd.DataFrame) -> tuple:
        """
        Simula split temporal basado en actividad académica
        
        Train: Estudiantes con primera actividad temprana
        Test: Estudiantes con primera actividad tardía
        """
        logger.info("🕒 Creating temporal split...")
        
        # Convertir fechas
        df['primera_actividad'] = pd.to_datetime(df['primera_actividad'])
        
        # Punto de corte temporal (70% datos más antiguos para train)
        cutoff_date = df['primera_actividad'].quantile(0.7)
        
        train_mask = df['primera_actividad'] <= cutoff_date
        test_mask = df['primera_actividad'] > cutoff_date
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        logger.info(f"   📊 Train period: hasta {cutoff_date.date()}")
        logger.info(f"   📊 Test period: después {cutoff_date.date()}")
        logger.info(f"   👥 Train students: {len(train_df)}")
        logger.info(f"   👥 Test students: {len(test_df)}")
        
        return train_df, test_df
    
    def validate_feature_distribution(self, train_df: pd.DataFrame, 
                                    test_df: pd.DataFrame) -> dict:
        """
        Validar que la distribución de features sea similar entre train/test
        """
        logger.info("📊 Validating feature distributions...")
        
        # Features clave para validar distribución
        key_features = ['average_grade', 'pct_low_combined', 'grade_std', 
                       'total_grades', 'subjects_count']
        
        distribution_stats = {}
        
        for feature in key_features:
            if feature in train_df.columns and feature in test_df.columns:
                # Convertir a numérico y manejar errores
                train_values = pd.to_numeric(train_df[feature], errors='coerce').dropna()
                test_values = pd.to_numeric(test_df[feature], errors='coerce').dropna()
                
                if len(train_values) == 0 or len(test_values) == 0:
                    logger.warning(f"   ⚠️ {feature}: No valid numeric values")
                    continue
                
                train_mean = train_values.mean()
                test_mean = test_values.mean()
                train_std = train_values.std()
                test_std = test_values.std()
                
                # Evitar división por cero
                if train_mean == 0 or train_std == 0:
                    continue
                
                # Calcular diferencia relativa
                mean_diff = abs(train_mean - test_mean) / abs(train_mean)
                std_diff = abs(train_std - test_std) / abs(train_std)
                
                distribution_stats[feature] = {
                    'train_mean': train_mean,
                    'test_mean': test_mean,
                    'train_std': train_std,
                    'test_std': test_std,
                    'mean_diff_pct': mean_diff * 100,
                    'std_diff_pct': std_diff * 100,
                    'distribution_stable': mean_diff < 0.1 and std_diff < 0.2
                }
                
                logger.info(f"   {feature}:")
                logger.info(f"     Mean diff: {mean_diff*100:.1f}%")
                logger.info(f"     Std diff: {std_diff*100:.1f}%")
                logger.info(f"     Stable: {distribution_stats[feature]['distribution_stable']}")
        
        return distribution_stats
    
    def cross_validate_temporal(self, df: pd.DataFrame, n_splits: int = 3) -> dict:
        """
        Validación cruzada temporal usando TimeSeriesSplit simulado
        """
        logger.info(f"🔄 Temporal cross-validation with {n_splits} splits...")
        
        # Definir features numéricas específicas (sin data leakage)
        numeric_features = [
            'edad', 'estrato', 'total_grades', 'average_grade', 'grade_std', 
            'min_grade', 'max_grade', 'subjects_count', 'grades_excellent', 
            'grades_good', 'grades_satisfactory', 'grades_low', 'grades_very_low',
            'meses_activos', 'grades_per_month', 'days_since_last_activity',
            'pct_excellent', 'pct_good', 'pct_satisfactory', 'pct_low_combined',
            'grade_range', 'grade_consistency'
        ]
        
        # Filtrar solo features que existen en el dataset
        feature_cols = [col for col in numeric_features if col in df.columns]
        
        logger.info(f"   📊 Using {len(feature_cols)} numeric features")
        
        # Preparar datos con manejo robusto de tipos
        X = df[feature_cols].copy()
        
        # Convertir a numérico y manejar errores
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Llenar NaN con median de cada columna
        X = X.fillna(X.median())
        
        # Crear target binario si no existe
        if 'is_high_risk' not in df.columns:
            if 'risk_level' in df.columns:
                y = df['risk_level'].isin(['ALTO', 'CRITICO']).astype(int)
            else:
                logger.error("❌ No target variable found (is_high_risk or risk_level)")
                return {}
        else:
            y = df['is_high_risk'].astype(int)
        
        # Ordenar por primera actividad para split temporal
        df_sorted = df.sort_values('primera_actividad').reset_index(drop=True)
        X_sorted = df_sorted[feature_cols].fillna(df_sorted[feature_cols].median())
        y_sorted = df_sorted['is_high_risk'].astype(int)
        
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            logger.info(f"   Fold {fold + 1}/{n_splits}...")
            
            X_train_fold = X_sorted.iloc[train_idx]
            X_test_fold = X_sorted.iloc[test_idx]
            y_train_fold = y_sorted.iloc[train_idx]
            y_test_fold = y_sorted.iloc[test_idx]
            
            # Entrenar modelo
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, 
                class_weight='balanced'
            )
            model.fit(X_train_fold, y_train_fold)
            
            # Predicción
            y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
            auc_score = roc_auc_score(y_test_fold, y_pred_proba)
            
            cv_scores.append(auc_score)
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'auc_score': auc_score,
                'train_risk_pct': y_train_fold.mean() * 100,
                'test_risk_pct': y_test_fold.mean() * 100
            })
            
            logger.info(f"     AUC: {auc_score:.3f}")
            logger.info(f"     Train risk%: {y_train_fold.mean()*100:.1f}%")
            logger.info(f"     Test risk%: {y_test_fold.mean()*100:.1f}%")
        
        results = {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'cv_auc_scores': cv_scores,
            'fold_details': fold_results
        }
        
        logger.info(f"✅ Temporal CV AUC: {results['cv_auc_mean']:.3f}±{results['cv_auc_std']:.3f}")
        
        return results
    
    def validate_business_logic(self, df: pd.DataFrame) -> dict:
        """
        Validar que las predicciones tengan sentido desde perspectiva educativa
        """
        logger.info("🎓 Validating business logic...")
        
        # Convertir columnas a numérico si es necesario
        df_clean = df.copy()
        for col in ['average_grade', 'pct_low_combined']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Crear target binario si no existe
        if 'is_high_risk' not in df_clean.columns:
            if 'risk_level' in df_clean.columns:
                df_clean['is_high_risk'] = df_clean['risk_level'].isin(['ALTO', 'CRITICO']).astype(int)
            else:
                logger.warning("⚠️ No target variable found for business logic validation")
                return {'business_logic_valid': False}
        
        business_validation = {}
        
        try:
            # 1. Estudiantes con promedio muy bajo deben tener alto riesgo
            if 'average_grade' in df_clean.columns:
                low_avg_students = df_clean[df_clean['average_grade'] < 2.5]
                high_risk_low_avg = low_avg_students['is_high_risk'].mean() if len(low_avg_students) > 0 else 0
            else:
                high_risk_low_avg = 0
            
            # 2. Estudiantes con muchas notas bajas deben tener alto riesgo  
            if 'pct_low_combined' in df_clean.columns:
                high_low_notes = df_clean[df_clean['pct_low_combined'] > 0.5]
                high_risk_low_notes = high_low_notes['is_high_risk'].mean() if len(high_low_notes) > 0 else 0
            else:
                high_risk_low_notes = 0
            
            # 3. Estudiantes excelentes deben tener bajo riesgo
            if 'average_grade' in df_clean.columns:
                excellent_students = df_clean[df_clean['average_grade'] > 4.2]
                low_risk_excellent = 1 - excellent_students['is_high_risk'].mean() if len(excellent_students) > 0 else 0
            else:
                low_risk_excellent = 0
            
            business_validation = {
                'low_avg_high_risk_pct': high_risk_low_avg * 100,
                'high_low_notes_high_risk_pct': high_risk_low_notes * 100,
                'excellent_low_risk_pct': low_risk_excellent * 100,
                'business_logic_valid': (
                    high_risk_low_avg > 0.7 and 
                    high_risk_low_notes > 0.7 and 
                    low_risk_excellent > 0.7
                )
            }
            
            logger.info(f"   📉 Low avg (< 2.5) → High risk: {high_risk_low_avg*100:.1f}%")
            logger.info(f"   📉 High low notes (> 50%) → High risk: {high_risk_low_notes*100:.1f}%")
            logger.info(f"   📈 Excellent (> 4.2) → Low risk: {low_risk_excellent*100:.1f}%")
            logger.info(f"   ✅ Business logic valid: {business_validation['business_logic_valid']}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error in business logic validation: {e}")
            business_validation = {'business_logic_valid': False}
        
        return business_validation
    
    def generate_validation_report(self, df: pd.DataFrame) -> dict:
        """
        Generar reporte completo de validación
        """
        logger.info("📋 Generating validation report...")
        
        # Crear target binario si no existe
        df_work = df.copy()
        if 'is_high_risk' not in df_work.columns:
            if 'risk_level' in df_work.columns:
                df_work['is_high_risk'] = df_work['risk_level'].isin(['ALTO', 'CRITICO']).astype(int)
            else:
                logger.warning("⚠️ No target variable found, creating dummy target")
                df_work['is_high_risk'] = 0
        
        report = {
            'dataset_info': {
                'total_students': len(df_work),
                'high_risk_pct': df_work['is_high_risk'].mean() * 100,
                'features_count': len([col for col in df_work.columns 
                                     if col not in ['student_id', 'nombre_completo', 'risk_level']])
            }
        }
        
        # Validación temporal
        train_df, test_df = self.simulate_temporal_split(df_work)
        report['temporal_split'] = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_risk_pct': train_df['is_high_risk'].mean() * 100,
            'test_risk_pct': test_df['is_high_risk'].mean() * 100
        }
        
        # Distribución de features
        report['feature_distribution'] = self.validate_feature_distribution(train_df, test_df)
        
        # Cross-validation temporal
        report['temporal_cv'] = self.cross_validate_temporal(df_work)
        
        # Validación de lógica de negocio
        report['business_logic'] = self.validate_business_logic(df_work)
        
        # Interpretación de resultados
        cv_auc = report['temporal_cv']['cv_auc_mean']
        
        if cv_auc > 0.95:
            interpretation = "⚠️ EXTREMELY HIGH - Possible overfitting or data leakage"
        elif cv_auc > 0.85:
            interpretation = "🟡 HIGH - Good but verify generalization"
        elif cv_auc > 0.75:
            interpretation = "✅ GOOD - Realistic and useful"
        elif cv_auc > 0.65:
            interpretation = "🟠 MODERATE - Needs improvement"
        else:
            interpretation = "❌ LOW - Model not useful"
        
        report['interpretation'] = {
            'auc_level': cv_auc,
            'assessment': interpretation,
            'production_ready': cv_auc > 0.75 and cv_auc < 0.95
        }
        
        return report

def main():
    """Ejecutar validación temporal completa"""
    logger.info("🚀 MARI AI - TEMPORAL MODEL VALIDATION")
    logger.info("=" * 60)
    
    # Cargar datos
    data_file = "data/ml/mari_ai_ml_dataset_main.csv"
    df = pd.read_csv(data_file)
    
    logger.info(f"📊 Dataset loaded: {len(df)} students")
    
    # Crear validador
    validator = TemporalModelValidator()
    
    # Ejecutar validación completa
    report = validator.generate_validation_report(df)
    
    # Mostrar resultados
    logger.info("\n" + "="*60)
    logger.info("📋 VALIDATION REPORT SUMMARY")
    logger.info("="*60)
    
    logger.info(f"📊 Dataset: {report['dataset_info']['total_students']} students")
    logger.info(f"⚠️ High risk: {report['dataset_info']['high_risk_pct']:.1f}%")
    
    logger.info(f"\n🕒 Temporal Split:")
    logger.info(f"   Train: {report['temporal_split']['train_size']} students")
    logger.info(f"   Test: {report['temporal_split']['test_size']} students")
    
    logger.info(f"\n🔄 Temporal CV Results:")
    cv_results = report['temporal_cv']
    logger.info(f"   AUC: {cv_results['cv_auc_mean']:.3f}±{cv_results['cv_auc_std']:.3f}")
    
    logger.info(f"\n🎓 Business Logic:")
    bl = report['business_logic']
    logger.info(f"   Valid: {bl['business_logic_valid']}")
    
    logger.info(f"\n🎯 FINAL ASSESSMENT:")
    interp = report['interpretation']
    logger.info(f"   AUC Level: {interp['auc_level']:.3f}")
    logger.info(f"   Assessment: {interp['assessment']}")
    logger.info(f"   Production Ready: {interp['production_ready']}")
    
    # Guardar reporte
    import json
    with open("data/ml/validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("\n💾 Validation report saved to: data/ml/validation_report.json")
    logger.info("\n✅ TEMPORAL VALIDATION COMPLETED!")

if __name__ == "__main__":
    main()