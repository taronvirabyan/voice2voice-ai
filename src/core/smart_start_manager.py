"""
Smart Start Manager - Оптимизация воспринимаемой скорости отклика
КРИТИЧНО: Безопасная реализация без нарушения основного потока
"""

import asyncio
import time
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field

from .logging import LoggerMixin


@dataclass
class SessionSmartStartState:
    """Состояние умного старта для сессии"""
    accumulated_words: List[str] = field(default_factory=list)
    thinking_indicator_sent: bool = False
    tts_prewarmed: bool = False
    first_words_timestamp: Optional[float] = None
    ai_processing_started: bool = False
    
    def reset(self):
        """Сброс состояния для новой фразы"""
        self.accumulated_words.clear()
        self.thinking_indicator_sent = False
        self.tts_prewarmed = False
        self.first_words_timestamp = None
        self.ai_processing_started = False


class SmartStartManager(LoggerMixin):
    """
    Менеджер умного старта для психологического ускорения отклика
    
    Принцип работы:
    1. Накапливаем первые слова пользователя
    2. При достижении порога (3+ слова) показываем "AI думает..."
    3. Параллельно прогреваем TTS для быстрого синтеза
    4. Создаем ощущение мгновенного отклика
    """
    
    def __init__(self, 
                 word_threshold: int = 3,
                 enable_thinking_indicator: bool = True,
                 enable_tts_prewarm: bool = True,
                 min_word_length: int = 2,
                 min_trigger_interval: float = 1.0,
                 confidence_threshold: float = 0.7):
        """
        Args:
            word_threshold: Минимальное количество слов для активации
            enable_thinking_indicator: Включить индикатор "AI думает"
            enable_tts_prewarm: Включить предварительный прогрев TTS
            min_word_length: Минимальная длина слова для учета (защита от "а", "и")
            min_trigger_interval: Минимальный интервал между активациями (сек)
            confidence_threshold: Минимальная уверенность для активации
        """
        self.word_threshold = word_threshold
        self.enable_thinking_indicator = enable_thinking_indicator
        self.enable_tts_prewarm = enable_tts_prewarm
        self.min_word_length = min_word_length
        self.min_trigger_interval = min_trigger_interval
        self.confidence_threshold = confidence_threshold
        
        # Состояния для каждой сессии
        self._session_states: Dict[str, SessionSmartStartState] = {}
        self._state_lock = asyncio.Lock()
        
        # Защита от слишком частых срабатываний
        self._last_trigger_time: Dict[str, float] = {}
        
        # Callbacks для действий
        self._thinking_indicator_callback: Optional[Callable] = None
        self._tts_prewarm_callback: Optional[Callable] = None
        
        self.logger.info(
            "SmartStartManager initialized",
            word_threshold=word_threshold,
            thinking_indicator=enable_thinking_indicator,
            tts_prewarm=enable_tts_prewarm
        )
    
    def set_thinking_indicator_callback(self, callback: Callable):
        """Установить callback для отправки индикатора размышления"""
        self._thinking_indicator_callback = callback
        
    def set_tts_prewarm_callback(self, callback: Callable):
        """Установить callback для прогрева TTS"""
        self._tts_prewarm_callback = callback
    
    async def get_or_create_session_state(self, session_id: str) -> SessionSmartStartState:
        """Получить или создать состояние сессии"""
        async with self._state_lock:
            if session_id not in self._session_states:
                self._session_states[session_id] = SessionSmartStartState()
            return self._session_states[session_id]
    
    async def process_transcript_text(self, session_id: str, text: str, confidence: float = 1.0) -> None:
        """
        Обработать новый текст транскрипции
        
        Args:
            session_id: ID сессии
            text: Новый транскрибированный текст
            confidence: Уверенность транскрипции (0-1)
        """
        try:
            state = await self.get_or_create_session_state(session_id)
            
            # Разбиваем текст на слова и фильтруем короткие
            words = [w for w in text.strip().split() if len(w) >= self.min_word_length]
            if not words:
                self.logger.debug(
                    "Smart Start: Skipping short/empty words",
                    session_id=session_id,
                    original_text=text[:50]
                )
                return
            
            # Проверяем уверенность транскрипции
            if confidence < self.confidence_threshold:
                self.logger.debug(
                    "Smart Start: Low confidence, skipping",
                    session_id=session_id,
                    confidence=confidence,
                    threshold=self.confidence_threshold
                )
                return
            
            # Если это первые слова - запоминаем время
            if not state.accumulated_words and not state.first_words_timestamp:
                state.first_words_timestamp = time.time()
                self.logger.info(
                    "Smart Start: First words detected",
                    session_id=session_id,
                    first_words=words[:3],
                    confidence=confidence
                )
            
            # Накапливаем слова
            state.accumulated_words.extend(words)
            current_word_count = len(state.accumulated_words)
            
            self.logger.debug(
                "Smart Start: Words accumulated",
                session_id=session_id,
                total_words=current_word_count,
                new_words=len(words)
            )
            
            # Проверяем достижение порога
            if (current_word_count >= self.word_threshold and 
                not state.thinking_indicator_sent and
                not state.ai_processing_started):
                
                # Защита от слишком частых срабатываний
                current_time = time.time()
                last_trigger = self._last_trigger_time.get(session_id, 0)
                
                if current_time - last_trigger < self.min_trigger_interval:
                    self.logger.debug(
                        "Smart Start: Skipping - too soon after last trigger",
                        session_id=session_id,
                        time_since_last=current_time - last_trigger,
                        min_interval=self.min_trigger_interval
                    )
                    return
                
                self.logger.info(
                    "Smart Start: Threshold reached, triggering optimizations",
                    session_id=session_id,
                    word_count=current_word_count,
                    meaningful_words=state.accumulated_words
                )
                
                # Обновляем время последнего срабатывания
                self._last_trigger_time[session_id] = current_time
                
                # Запускаем оптимизации параллельно
                tasks = []
                
                # 1. Показываем индикатор размышления
                if self.enable_thinking_indicator and self._thinking_indicator_callback:
                    tasks.append(self._send_thinking_indicator(session_id, state))
                
                # 2. Прогреваем TTS
                if self.enable_tts_prewarm and self._tts_prewarm_callback:
                    tasks.append(self._prewarm_tts(session_id, state))
                
                # Выполняем параллельно
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
        except Exception as e:
            self.logger.error(
                "Error in Smart Start processing",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            # НЕ прерываем основной поток при ошибке!
    
    async def _send_thinking_indicator(self, session_id: str, state: SessionSmartStartState):
        """Отправить индикатор размышления"""
        try:
            if state.thinking_indicator_sent:
                return
                
            await self._thinking_indicator_callback(session_id)
            state.thinking_indicator_sent = True
            
            self.logger.info(
                "Smart Start: Thinking indicator sent",
                session_id=session_id,
                latency_ms=int((time.time() - state.first_words_timestamp) * 1000)
            )
            
        except Exception as e:
            self.logger.warning(
                "Failed to send thinking indicator",
                session_id=session_id,
                error=str(e)
            )
    
    async def _prewarm_tts(self, session_id: str, state: SessionSmartStartState):
        """Прогреть TTS сервис"""
        try:
            if state.tts_prewarmed:
                return
                
            await self._tts_prewarm_callback(session_id)
            state.tts_prewarmed = True
            
            self.logger.info(
                "Smart Start: TTS prewarmed",
                session_id=session_id
            )
            
        except Exception as e:
            self.logger.warning(
                "Failed to prewarm TTS",
                session_id=session_id,
                error=str(e)
            )
    
    async def mark_ai_processing_started(self, session_id: str):
        """Отметить начало обработки AI"""
        try:
            state = await self.get_or_create_session_state(session_id)
            state.ai_processing_started = True
            
            if state.first_words_timestamp:
                latency = (time.time() - state.first_words_timestamp) * 1000
                self.logger.info(
                    "Smart Start: AI processing started",
                    session_id=session_id,
                    total_latency_ms=int(latency),
                    words_accumulated=len(state.accumulated_words)
                )
                
        except Exception as e:
            self.logger.error(
                "Error marking AI processing",
                session_id=session_id,
                error=str(e)
            )
    
    async def reset_session_state(self, session_id: str):
        """Сбросить состояние сессии для новой фразы"""
        try:
            state = await self.get_or_create_session_state(session_id)
            state.reset()
            
            self.logger.debug(
                "Smart Start: Session state reset",
                session_id=session_id
            )
            
        except Exception as e:
            self.logger.error(
                "Error resetting session state",
                session_id=session_id,
                error=str(e)
            )
    
    async def cleanup_session(self, session_id: str):
        """Очистить данные сессии"""
        async with self._state_lock:
            if session_id in self._session_states:
                del self._session_states[session_id]
            if session_id in self._last_trigger_time:
                del self._last_trigger_time[session_id]
            self.logger.debug(
                "Smart Start: Session cleaned up",
                session_id=session_id
            )
    
    def get_stats(self) -> Dict:
        """Получить статистику работы"""
        return {
            "active_sessions": len(self._session_states),
            "word_threshold": self.word_threshold,
            "thinking_indicator_enabled": self.enable_thinking_indicator,
            "tts_prewarm_enabled": self.enable_tts_prewarm
        }