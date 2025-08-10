# mari_ai_agent/app/utils/exceptions.py
"""
Excepciones personalizadas para Mari AI
"""

class MariAIException(Exception):
    """Excepción base para Mari AI"""
    pass

class StudentNotFoundError(MariAIException):
    """Error cuando no se encuentra un estudiante"""
    pass

class DataValidationError(MariAIException):
    """Error de validación de datos"""
    pass

class DatabaseConnectionError(MariAIException):
    """Error de conexión a la base de datos"""
    pass

class InvalidParameterError(MariAIException):
    """Error de parámetro inválido"""
    pass

class AuthenticationError(MariAIException):
    """Error de autenticación"""
    pass

class AuthorizationError(MariAIException):
    """Error de autorización"""
    pass

class ExternalServiceError(MariAIException):
    """Error de servicio externo"""
    pass