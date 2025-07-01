"""
Централизованная конфигурация приложения
КРИТИЧЕСКИ ВАЖНО: Все настройки должны быть валидными
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Настройки приложения с валидацией"""
    
    # ===== API КЛЮЧИ (ОБЯЗАТЕЛЬНЫЕ) =====
    salute_client_id: str
    salute_client_secret: str
    gemini_api_key: str
    
    # ===== РЕЗЕРВНЫЕ GEMINI КЛЮЧИ =====
    gemini_api_key_2: Optional[str] = "AIzaSyD2KsotVLNS_8MAEJqCVzbxAh8U2i-xYPs"
    gemini_api_key_3: Optional[str] = "AIzaSyCZcwunnoBBqhTYtbgJZl-7hpqJEVqFTeY"
    
    # ===== VOICE2VOICE API КЛЮЧИ =====
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    
    # ===== WHISPER НАСТРОЙКИ =====
    whisper_model: str = "large-v3"  # tiny, base, small, medium, large, large-v2, large-v3
    whisper_language: str = "ru"
    whisper_chunk_duration: int = 30  # секунды
    whisper_overlap_duration: int = 5  # секунды
    
    # ===== REDIS НАСТРОЙКИ =====
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_timeout: int = 5
    
    # ===== СЕРВЕР НАСТРОЙКИ =====
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    debug: bool = False
    cors_origins: List[str] = ["*"]
    
    # ===== АУДИО НАСТРОЙКИ =====
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_format: str = "pcm16"
    max_audio_length: int = 300  # секунды
    
    # ===== AI НАСТРОЙКИ =====
    gemini_model: str = "gemini-2.5-flash"  # Новейшая модель с улучшенной производительностью
    moderator_model: str = "gemini-2.5-flash"  # Используем ту же модель для модерации
    max_tokens: int = 150
    temperature: float = 0.7
    moderator_temperature: float = 0.3
    
    # ===== ПРОИЗВОДИТЕЛЬНОСТЬ =====
    max_concurrent_sessions: int = 10
    session_timeout: int = 300  # 5 минут
    request_timeout: int = 30
    max_history_length: int = 20
    
    # ===== VAD (Voice Activity Detection) =====
    enable_vad: bool = False  # Включить VAD для оптимизации распознавания
    
    # ===== I/O Оптимизация =====
    enable_ram_tempfiles: bool = False  # Использовать RAM для временных файлов (осторожно!)
    ram_tempfiles_max_size: int = 100  # MB - максимальный размер RAM для временных файлов
    
    # ===== SBER SALUTE SPEECH =====
    salute_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    salute_stt_url: str = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    salute_tts_url: str = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    salute_voice: str = "Nec_24000"  # Мужской голос
    
    # ===== ЛОГИРОВАНИЕ И МОНИТОРИНГ =====
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    health_check_interval: int = 30
    
    # ===== ПРОМПТЫ ПО УМОЛЧАНИЮ =====
    default_prompt: str = "Узнай собеседника получше. Задавай дружелюбные вопросы о его интересах, хобби и жизни."
    
    # Промпты для переключения
    dog_prompt: str = """Отлично, что у тебя есть собака! Какая порода? Кстати, недавно видел классные штуки для собак - умные ошейники с GPS, автокормушки, игрушки-головоломки. У моего друга собака в восторге от интерактивной игрушки, которая сама катается и развлекает питомца, пока хозяин на работе."""
    money_prompt: str = """Понимаю, с деньгами сейчас у многих напряженка. Кстати, знаешь что помогло моим знакомым? Кто-то начал продавать ненужные вещи на Авито, кто-то освоил новую профессию онлайн, а кто-то нашел подработку в интернете. Есть классные курсы и платформы для дополнительного заработка. Могу поделиться, если интересно."""
    children_prompt: str = """Здорово, что у тебя есть дети! В каком они возрасте? Сейчас столько интересного для развития детей - от обучающих приложений и онлайн-курсов до развивающих игрушек и детских смарт-часов. У знакомых дочка в восторге от интерактивного глобуса с дополненной реальностью - география стала любимым предметом!"""
    
    # Универсальный адаптивный промпт
    sales_prompt: str = """Ты умный собеседник, который предлагает решения под ТЕКУЩИЙ контекст разговора.

🔴 КРИТИЧЕСКИ ВАЖНО: Это ДИНАМИЧЕСКИЙ промпт! Анализируй КАЖДОЕ новое сообщение и адаптируйся!

СИСТЕМА АДАПТАЦИИ - реагируй на ПОСЛЕДНЕЕ сообщение пользователя:
• "нет денег", "дорого", "финансовые проблемы" → ПЕРЕКЛЮЧИСЬ на способы заработка, фриланс, курсы
• "кошка", "собака", "питомец" → ПЕРЕКЛЮЧИСЬ на товары для животных, корма, игрушки  
• "устал", "стресс", "тяжело" → ПЕРЕКЛЮЧИСЬ на поддержку, релаксацию, психологическую помощь
• "хобби", "интересы" → ПЕРЕКЛЮЧИСЬ на товары и курсы по теме хобби
• "дети" → ПЕРЕКЛЮЧИСЬ на детские товары, развивающие игры
• "работа", "карьера" → ПЕРЕКЛЮЧИСЬ на курсы повышения квалификации
• "здоровье", "болит" → ПЕРЕКЛЮЧИСЬ на медицинские услуги, витамины
• Любая новая тема → АДАПТИРУЙСЯ под неё!

ЗОЛОТОЕ ПРАВИЛО: Если пользователь сменил тему - ты ОБЯЗАН сменить фокус предложений!

ПРАВИЛА:
1. Анализируй ИМЕННО ПОСЛЕДНЕЕ сообщение, не застревай на старых темах
2. При смене темы - НЕМЕДЛЕННО меняй направление предложений
3. Подавай как личный опыт: "У меня был похожий случай...", "Мой друг решил это так..."
4. Будь искренним и не навязчивым
5. НЕ СМЕШИВАЙ темы - фокусируйся на текущей"""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def validate_required_settings(self) -> None:
        """Проверка критически важных настроек"""
        if not self.salute_client_id or self.salute_client_id == "your_client_id_here":
            raise ValueError("SALUTE_CLIENT_ID must be set!")
            
        if not self.salute_client_secret or self.salute_client_secret == "your_client_secret_here":
            raise ValueError("SALUTE_CLIENT_SECRET must be set!")
            
        if not self.gemini_api_key or self.gemini_api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY must be set!")


@lru_cache()
def get_settings() -> Settings:
    """Singleton для настроек с кешированием"""
    settings = Settings()
    settings.validate_required_settings()
    return settings


# Глобальный экземпляр настроек
settings = get_settings()