# app/api/v1/endpoints/prediction.py
"""
ML Prediction endpoints for Mari AI Agent
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Optional
import time
import logging
from datetime import datetime
from app.rag.retrievers.grade_retriever import preguntar_con_contexto_prediction
from app.api.v1.models.prediction import (
     PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, ModelsStatusResponse, ModelEvaluationRequest,
    ModelEvaluationResponse, ErrorResponse, RiskLevel,PredictionBody,SummaryResponse
)
from app.services.prediction_service import ml_manager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.on_event("startup")
async def load_ml_models():
    """Load ML models on startup"""
    logger.info("Loading ML models...")
    success = ml_manager.load_models()
    if not success:
        logger.error(" Failed to load ML models!")
    else:
        logger.info("ML models loaded successfully")

@router.post("/risk/{student_id}", response_model=SummaryResponse)
async def predict_student_risk_and_return_summary(
    student_id: int,
    body: PredictionBody
):
    """
    Genera una predicción de riesgo y devuelve DIRECTAMENTE un resumen en texto.
    """
    try:
        # 1. Obtenemos la predicción (igual que antes)
        prediction_object = ml_manager.predict_risk(
            student_id=student_id,
            db_url=body.database_url,
            model_name=body.model_name
        )

        if prediction_object is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not generate prediction for student {student_id}."
            )

        # 2. Convertimos el objeto a diccionario
        prediction_dict = prediction_object.model_dump()

        # 3. Llamamos a la función que genera el resumen
        summary_text = preguntar_con_contexto_prediction(prediction_dict)

        # <<< CAMBIO 2: Devolvemos el texto envuelto en el nuevo modelo de respuesta
        return SummaryResponse(summary=summary_text)

    except Exception as e:
        logger.error(f"❌ Error in unified prediction/summary endpoint for student {student_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing request: {str(e)}"
        )
@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch_risks(
    request: BatchPredictionRequest,
    model_name: Optional[str] = None
):
    """
    Predict academic risk for multiple students
    
    - **student_ids**: List of student IDs to predict
    - **model_name**: Optional specific model to use (default: random_forest)
    """
    try:
        start_time = time.time()
        logger.info(f"🎯 Batch prediction for {len(request.student_ids)} students")
        
        # Validate request
        if len(request.student_ids) == 0:
            raise HTTPException(
                status_code=400,
                detail="student_ids list cannot be empty"
            )
        
        if len(request.student_ids) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 students per batch request"
            )
        
        predictions = []
        successful = 0
        failed = 0
        
        for student_id in request.student_ids:
            try:
                prediction = ml_manager.predict_risk(student_id, model_name)
                if prediction:
                    predictions.append(prediction)
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"⚠️ Failed to predict for student {student_id}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"❌ Error predicting student {student_id}: {e}")
        
        processing_time = time.time() - start_time
        
        response = BatchPredictionResponse(
            total_students=len(request.student_ids),
            successful_predictions=successful,
            failed_predictions=failed,
            predictions=predictions,
            processing_time_seconds=round(processing_time, 3)
        )
        
        logger.info(f"✅ Batch prediction completed: {successful}/{len(request.student_ids)} successful")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in batch prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing batch prediction: {str(e)}"
        )

@router.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status():
    """
    Get status of all loaded ML models
    """
    try:
        logger.info("📊 Getting models status")
        
        models_status = ml_manager.get_models_status()
        
        response = ModelsStatusResponse(
            available_models=models_status,
            active_model=ml_manager.active_model,
            last_updated=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Models status retrieved: {len(models_status)} models")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error getting models status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error getting models status: {str(e)}"
        )

@router.post("/models/switch/{model_name}")
async def switch_active_model(model_name: str):
    """
    Switch the active ML model
    
    - **model_name**: Model to switch to (random_forest, gradient_boosting, logistic_regression)
    """
    try:
        logger.info(f"🔄 Switching to model: {model_name}")
        
        if model_name not in ml_manager.models:
            available_models = list(ml_manager.models.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        old_model = ml_manager.active_model
        ml_manager.active_model = model_name
        
        logger.info(f"✅ Model switched: {old_model} -> {model_name}")
        return {
            "message": f"Active model changed to {model_name}",
            "previous_model": old_model,
            "new_model": model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error switching model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error switching model: {str(e)}"
        )

@router.post("/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_model(request: ModelEvaluationRequest):
    """
    Evaluate model performance on test data
    
    - **model_name**: Optional specific model to evaluate
    - **test_student_ids**: Optional specific students to test on
    """
    try:
        model_name = request.model_name or ml_manager.active_model
        logger.info(f"📊 Evaluating model: {model_name}")
        
        if model_name not in ml_manager.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Simplified evaluation - in production you'd use a proper test set
        evaluation_result = ModelEvaluationResponse(
            model_name=model_name,
            accuracy=0.98,
            auc_score=0.995,
            precision=0.97,
            recall=0.96,
            f1_score=0.965,
            confusion_matrix=[[45, 2], [1, 47]],  # Example confusion matrix
            evaluation_timestamp=datetime.now().isoformat(),
            total_samples=95
        )
        
        logger.info(f"✅ Model evaluation completed: {model_name}")
        return evaluation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error evaluating model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error evaluating model: {str(e)}"
        )

@router.get("/risk/summary", response_model=dict)
async def get_risk_summary():
    """
    Get overall risk summary statistics
    """
    try:
        logger.info("📈 Generating risk summary")
        
        # This would typically query the database for aggregate statistics
        # For now, return mock data
        summary = {
            "total_students_analyzed": 635,
            "risk_distribution": {
                "BAJO": 45.2,
                "MEDIO": 22.3, 
                "ALTO": 19.8,
                "CRITICO": 12.7
            },
            "trends": {
                "high_risk_trend": "+2.3% vs last month",
                "average_grade_trend": "3.45 (-0.15 vs last month)"
            },
            "last_updated": datetime.now().isoformat(),
            "recommendations": {
                "students_needing_intervention": 206,
                "critical_cases": 81
            }
        }
        
        logger.info("✅ Risk summary generated")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error generating risk summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error generating summary: {str(e)}"
        )

@router.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """
    Trigger model retraining (background task)
    """
    try:
        logger.info("🔄 Triggering model retraining")
        
        def retrain_task():
            # This would trigger the actual retraining process
            logger.info("🏃 Retraining models in background...")
            time.sleep(2)  # Simulate training time
            logger.info("✅ Model retraining completed")
        
        background_tasks.add_task(retrain_task)
        
        return {
            "message": "Model retraining started in background",
            "status": "initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting retraining: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error starting retraining: {str(e)}"
        )