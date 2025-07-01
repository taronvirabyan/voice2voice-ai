"""
Кастомные исключения для Voice2Voice системы
ВАЖНО: Все ошибки должны быть обработаны корректно
"""

from typing import Optional, Dict, Any


class Voice2VoiceException(Exception):
    """Базовое исключение для всех ошибок системы"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(Voice2VoiceException):
    """Ошибки конфигурации"""
    
    def __init__(self, message: str, missing_setting: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details={"missing_setting": missing_setting}
        )


class ServiceInitializationError(Voice2VoiceException):
    """Ошибки инициализации сервисов"""
    
    def __init__(self, service_name: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message=f"{service_name} initialization failed: {message}",
            error_code="SERVICE_INIT_ERROR",
            details={
                "service_name": service_name,
                "original_error": str(original_error) if original_error else None
            }
        )


class APIConnectionError(Voice2VoiceException):
    """Ошибки подключения к внешним API"""
    
    def __init__(
        self, 
        service_name: str, 
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__(
            message=f"{service_name}: {message}",
            error_code="API_CONNECTION_ERROR",
            details={
                "service": service_name,
                "status_code": status_code,
                "response_body": response_body
            }
        )


class SaluteSpeechError(APIConnectionError):
    """Ошибки Sber SaluteSpeech API"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(service_name="SaluteSpeech", message=message, **kwargs)


class ClaudeAPIError(APIConnectionError):
    """Ошибки Claude API"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(service_name="Claude", message=message, **kwargs)


class GeminiError(APIConnectionError):
    """Ошибки Gemini API"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(service_name="Gemini", message=message, **kwargs)


class AudioProcessingError(Voice2VoiceException):
    """Ошибки обработки аудио"""
    
    def __init__(self, message: str, audio_format: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            details={"audio_format": audio_format}
        )


class SessionError(Voice2VoiceException):
    """Ошибки управления сессиями"""
    
    def __init__(self, message: str, session_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="SESSION_ERROR",
            details={"session_id": session_id}
        )


class TranscriptionError(Voice2VoiceException):
    """Ошибки транскрипции"""
    
    def __init__(self, message: str, confidence: Optional[float] = None):
        super().__init__(
            message=message,
            error_code="TRANSCRIPTION_ERROR",
            details={"confidence": confidence}
        )


class TTSError(Voice2VoiceException):
    """Ошибки синтеза речи"""
    
    def __init__(self, message: str, text_length: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="TTS_ERROR",
            details={"text_length": text_length}
        )


class STTError(Voice2VoiceException):
    """Ошибки распознавания речи"""
    
    def __init__(self, message: str, audio_length: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="STT_ERROR",
            details={"audio_length": audio_length}
        )


class RedisConnectionError(Voice2VoiceException):
    """Ошибки подключения к Redis"""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="REDIS_CONNECTION_ERROR"
        )


class RedisError(Voice2VoiceException):
    """Общие ошибки Redis операций"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="REDIS_ERROR",
            details={"operation": operation}
        )


class RateLimitError(Voice2VoiceException):
    """Ошибки превышения лимитов"""
    
    def __init__(self, service: str, limit_type: str, reset_time: Optional[int] = None):
        message = f"Rate limit exceeded for {service} ({limit_type})"
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details={
                "service": service,
                "limit_type": limit_type,
                "reset_time": reset_time
            }
        )


class ValidationError(Voice2VoiceException):
    """Ошибки валидации данных"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": str(value) if value is not None else None}
        )


class WebSocketError(Voice2VoiceException):
    """Ошибки WebSocket соединения"""
    
    def __init__(self, message: str, connection_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_ERROR",
            details={"connection_id": connection_id}
        )


# Утилитарные функции для обработки ошибок

def handle_api_error(
    service_name: str,
    status_code: int,
    response_text: str,
    operation: str = "request"
) -> Voice2VoiceException:
    """Создание соответствующего исключения на основе ответа API"""
    
    if status_code == 401:
        return APIConnectionError(
            service_name=service_name,
            message=f"Authentication failed for {operation}",
            status_code=status_code,
            response_body=response_text
        )
    elif status_code == 429:
        return RateLimitError(
            service=service_name,
            limit_type="API calls"
        )
    elif status_code >= 500:
        return APIConnectionError(
            service_name=service_name,
            message=f"Server error during {operation}",
            status_code=status_code,
            response_body=response_text
        )
    else:
        return APIConnectionError(
            service_name=service_name,
            message=f"API error during {operation}",
            status_code=status_code,
            response_body=response_text
        )