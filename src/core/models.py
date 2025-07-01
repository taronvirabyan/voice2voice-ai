"""
Модели данных для Voice2Voice системы
Используем Pydantic для валидации и сериализации
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import time
import uuid


class SessionState(str, Enum):
    """Состояния сессии"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class MessageRole(str, Enum):
    """Роли в диалоге"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AudioChunk(BaseModel):
    """Чанк аудио данных для обработки"""
    session_id: str
    data: bytes
    timestamp: float = Field(default_factory=time.time)
    sequence_number: int
    is_final: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class TranscriptSegment(BaseModel):
    """Сегмент транскрипта диалога"""
    text: str = Field(min_length=1, max_length=1000)
    timestamp: float = Field(default_factory=time.time)
    speaker: MessageRole
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    duration: Optional[float] = None
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class PromptUpdate(BaseModel):
    """Обновление промпта от модератора"""
    session_id: str
    old_prompt: str
    new_prompt: str = Field(min_length=10, max_length=2000)  # Увеличен лимит для длинных промптов
    trigger_keywords: List[str] = []
    trigger_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)
    
    @validator('new_prompt')
    def validate_prompt(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Prompt too short')
        return v.strip()


class VoiceResponse(BaseModel):
    """Ответ AI для синтеза речи"""
    session_id: str
    text: str = Field(min_length=1, max_length=500)
    voice_config: Dict[str, Any] = Field(default_factory=lambda: {
        "voice": "Nec_24000",
        "sample_rate": 24000,
        "format": "pcm16"
    })
    priority: int = Field(default=5, ge=1, le=10)
    timestamp: float = Field(default_factory=time.time)


class SessionMetrics(BaseModel):
    """Метрики сессии для мониторинга"""
    session_id: str
    total_user_messages: int = 0
    total_ai_responses: int = 0
    prompt_changes: int = 0
    average_response_time: float = 0.0
    total_audio_duration: float = 0.0
    errors_count: int = 0
    last_activity: float = Field(default_factory=time.time)


class Session(BaseModel):
    """Полная информация о сессии диалога"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.ACTIVE
    
    # Текущий промпт и история
    current_prompt: str = "Узнай собеседника получше"
    transcript_history: List[TranscriptSegment] = []
    prompt_history: List[PromptUpdate] = []
    
    # Метаданные
    client_info: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Метрики
    metrics: SessionMetrics = Field(default_factory=lambda: SessionMetrics(session_id=""))
    
    def __init__(self, **data):
        super().__init__(**data)
        # Устанавливаем session_id для метрик
        if self.metrics.session_id == "":
            self.metrics.session_id = self.id
    
    def add_transcript(self, segment: TranscriptSegment) -> None:
        """Добавление сегмента транскрипта"""
        self.transcript_history.append(segment)
        self.updated_at = datetime.now()
        
        # Обновляем метрики
        if segment.speaker == MessageRole.USER:
            self.metrics.total_user_messages += 1
        elif segment.speaker == MessageRole.ASSISTANT:
            self.metrics.total_ai_responses += 1
            
        self.metrics.last_activity = time.time()
        
        # Ограничиваем историю
        max_history = 50
        if len(self.transcript_history) > max_history:
            self.transcript_history = self.transcript_history[-max_history:]
    
    def update_prompt(self, update: PromptUpdate) -> None:
        """Обновление промпта"""
        update.session_id = self.id
        self.current_prompt = update.new_prompt
        self.prompt_history.append(update)
        self.updated_at = datetime.now()
        self.metrics.prompt_changes += 1
    
    def get_recent_history(self, count: int = 10) -> List[TranscriptSegment]:
        """Получение последних N сообщений"""
        return self.transcript_history[-count:] if self.transcript_history else []
    
    def is_active(self) -> bool:
        """Проверка активности сессии"""
        return self.state == SessionState.ACTIVE
    
    def get_conversation_context(self) -> str:
        """Получение контекста диалога как строки"""
        if not self.transcript_history:
            return ""
            
        recent = self.get_recent_history(10)
        return "\n".join([f"{seg.speaker.value}: {seg.text}" for seg in recent])


class SystemHealth(BaseModel):
    """Состояние системы для мониторинга"""
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Статус сервисов
    services: Dict[str, bool] = Field(default_factory=dict)
    
    # Метрики производительности
    active_sessions: int = 0
    total_sessions: int = 0
    average_latency: float = 0.0
    
    # Ошибки
    errors_last_hour: int = 0
    last_error: Optional[str] = None
    
    def update_service_status(self, service_name: str, is_healthy: bool) -> None:
        """Обновление статуса сервиса"""
        self.services[service_name] = is_healthy
        self.timestamp = datetime.now()
        
        # Определяем общий статус
        if all(self.services.values()):
            self.status = "healthy"
        elif any(self.services.values()):
            self.status = "degraded"
        else:
            self.status = "unhealthy"


class APIError(BaseModel):
    """Модель для ошибок API"""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    trace_id: Optional[str] = None