class MariAIException(Exception):
    """Excepción base para Mari AI"""
    pass

class StudentNotFoundError(MariAIException):
    """Estudiante no encontrado"""
    pass

class DataValidationError(MariAIException):
    """Error de validación de datos"""
    pass

class DatabaseConnectionError(MariAIException):
    """Error de conexión a base de datos"""
    pass

class PredictionModelError(MariAIException):
    """Error en modelo de predicción"""
    pass

class RecommendationError(MariAIException):
    """Error en sistema de recomendaciones"""
    pass