"""
Настройка структурированного логирования
КРИТИЧНО для отладки и мониторинга production системы
"""

import structlog
import logging
import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from .config import settings


def setup_logging() -> None:
    """Настройка структурированного логирования"""
    
    # Создаем директорию для логов
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Процессоры для structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Добавляем JSON форматтер для production
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Настройка structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Настройка стандартного логгера Python
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Настройка файлового логгера
    file_handler = logging.FileHandler(
        logs_dir / "voice2voice.log",
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Форматтер для файлов
    if settings.log_format == "json":
        file_formatter = logging.Formatter('%(message)s')
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    
    # Добавляем к root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Настройка логгеров внешних библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Получение логгера с контекстом"""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


class LoggerMixin:
    """Миксин для добавления логгера в классы"""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Логгер с именем класса"""
        return get_logger(self.__class__.__name__)


def log_request(
    operation: str,
    service: str,
    session_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Создание контекста для логирования запроса"""
    context = {
        "operation": operation,
        "service": service,
        "timestamp": datetime.now().isoformat(),
    }
    
    if session_id:
        context["session_id"] = session_id
        
    context.update(kwargs)
    return context


def log_error(
    error: Exception,
    operation: str,
    service: str = None,
    session_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Создание контекста для логирования ошибки"""
    context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
    }
    
    if service:
        context["service"] = service
        
    if session_id:
        context["session_id"] = session_id
        
    # Добавляем детали ошибки если это наша кастомная ошибка
    if hasattr(error, 'error_code'):
        context["error_code"] = error.error_code
        
    if hasattr(error, 'details'):
        context["error_details"] = error.details
        
    context.update(kwargs)
    return context


def log_performance(
    operation: str,
    duration_ms: float,
    service: str = None,
    session_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Создание контекста для логирования производительности"""
    context = {
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "timestamp": datetime.now().isoformat(),
    }
    
    if service:
        context["service"] = service
        
    if session_id:
        context["session_id"] = session_id
        
    context.update(kwargs)
    return context