"""
Voice Activity Detector - Мгновенное определение начала речи
КРИТИЧНО: Для психологического ускорения отклика
"""

import asyncio
import time
from typing import Optional, Callable
import numpy as np
from dataclasses import dataclass

from .logging import LoggerMixin


@dataclass 
class VADState:
    """Состояние детектора голоса для сессии"""
    is_speaking: bool = False
    speech_start_time: Optional[float] = None
    silence_start_time: Optional[float] = None
    thinking_sent: bool = False
    thinking_sent_timestamp: Optional[float] = None  # Для предотвращения дубликатов
    audio_energy_history: list = None
    recording_start_time: Optional[float] = None  # Время начала записи
    chunk_count: int = 0  # Счетчик чанков
    
    def __post_init__(self):
        if self.audio_energy_history is None:
            self.audio_energy_history = []


class VoiceActivityDetector(LoggerMixin):
    """
    Детектор голосовой активности для мгновенного отклика
    
    Определяет начало речи по энергии аудио сигнала БЕЗ транскрипции
    Позволяет показать "AI думает" сразу при начале речи
    """
    
    def __init__(self,
                 energy_threshold: float = 0.01,
                 speech_start_chunks: int = 2,
                 silence_chunks: int = 10,
                 sample_rate: int = 16000):
        """
        Args:
            energy_threshold: Порог энергии для определения речи
            speech_start_chunks: Количество чанков для подтверждения начала речи
            silence_chunks: Количество тихих чанков для определения конца речи
            sample_rate: Частота дискретизации
        """
        self.energy_threshold = energy_threshold
        self.speech_start_chunks = speech_start_chunks
        self.silence_chunks = silence_chunks
        self.sample_rate = sample_rate
        
        # Состояния для каждой сессии
        self._session_states = {}
        self._state_lock = asyncio.Lock()
        
        # Callback для уведомления о начале речи
        self._speech_started_callback: Optional[Callable] = None
        
        self.logger.info(
            "VoiceActivityDetector initialized",
            energy_threshold=energy_threshold,
            speech_start_chunks=speech_start_chunks
        )
    
    def set_speech_started_callback(self, callback: Callable):
        """Установить callback для уведомления о начале речи"""
        self._speech_started_callback = callback
    
    async def get_or_create_state(self, session_id: str) -> VADState:
        """Получить или создать состояние для сессии"""
        async with self._state_lock:
            if session_id not in self._session_states:
                self._session_states[session_id] = VADState()
            return self._session_states[session_id]
    
    def calculate_audio_energy(self, audio_chunk: bytes) -> float:
        """
        Вычислить энергию аудио чанка
        
        Args:
            audio_chunk: Байты аудио (WebM или PCM)
            
        Returns:
            float: Нормализованная энергия (0-1)
        """
        try:
            chunk_size = len(audio_chunk)
            
            if chunk_size < 50:
                # Слишком маленький чанк - вероятно тишина
                return 0.0
            
            # Проверка на WebM/Opus заголовки
            if chunk_size > 4:
                # WebM начинается с EBML заголовка 0x1A45DFA3
                if audio_chunk[:4] == b'\x1a\x45\xdf\xa3':
                    self.logger.debug("Detected WebM EBML header, returning 0 energy")
                    return 0.0
                
                # Проверка на Opus заголовки в первых 50 байтах
                if b'Opus' in audio_chunk[:min(50, chunk_size)]:
                    self.logger.debug("Detected Opus header, returning 0 energy")
                    return 0.0
                
                # Проверка на другие WebM сигнатуры
                if b'webm' in audio_chunk[:min(50, chunk_size)].lower():
                    self.logger.debug("Detected WebM signature, returning 0 energy")
                    return 0.0
            
            # Анализируем энтропию данных
            # Больше вариативность = больше вероятность речи
            if chunk_size > 0:
                # Считаем уникальные байты
                unique_bytes = len(set(audio_chunk[:min(100, chunk_size)]))
                
                # Нормализуем: много уникальных байтов = высокая энергия
                # WebM с речью имеет высокую энтропию
                energy = unique_bytes / 100.0
                
                # Дополнительно учитываем размер чанка
                # Большие чанки часто содержат больше аудио данных
                size_factor = min(chunk_size / 1000.0, 1.0)
                
                # Комбинированная метрика
                combined_energy = (energy * 0.7 + size_factor * 0.3)
                
                return float(min(combined_energy, 1.0))
            
        except Exception as e:
            self.logger.warning(
                "Failed to calculate audio energy",
                error=str(e)
            )
            return 0.0
    
    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> bool:
        """
        Обработать аудио чанк и определить активность голоса
        
        Args:
            session_id: ID сессии
            audio_chunk: Байты аудио
            
        Returns:
            bool: True если обнаружено начало речи
        """
        try:
            state = await self.get_or_create_state(session_id)
            
            # Инициализируем время начала записи
            if state.recording_start_time is None:
                state.recording_start_time = time.time()
                self.logger.debug(
                    "VAD: Recording started",
                    session_id=session_id
                )
            
            # Увеличиваем счетчик чанков
            state.chunk_count += 1
            
            # Пропускаем первые несколько чанков (обычно содержат заголовки)
            if state.chunk_count <= 3:
                self.logger.debug(
                    f"VAD: Skipping initial chunk #{state.chunk_count} (likely headers)",
                    session_id=session_id,
                    chunk_size=len(audio_chunk)
                )
                return False
            
            # Не позволяем срабатывать в первую секунду записи
            time_since_start = time.time() - state.recording_start_time
            if time_since_start < 1.0:
                self.logger.debug(
                    f"VAD: Too early, only {time_since_start:.2f}s since recording start",
                    session_id=session_id
                )
                return False
            
            # Предотвращаем дубликаты сообщений
            current_time = time.time()
            if state.thinking_sent_timestamp:
                time_since_last = current_time - state.thinking_sent_timestamp
                if time_since_last < 3.0:  # Не отправляем чаще чем раз в 3 секунды
                    self.logger.debug(
                        f"VAD: Skipping duplicate, only {time_since_last:.2f}s since last message",
                        session_id=session_id
                    )
                    return False
            
            # Вычисляем энергию
            energy = self.calculate_audio_energy(audio_chunk)
            
            # Сохраняем историю для отладки
            state.audio_energy_history.append(energy)
            if len(state.audio_energy_history) > 100:  # Храним последние 100
                state.audio_energy_history.pop(0)
            
            # Определяем есть ли речь
            is_speech = energy > self.energy_threshold
            
            if is_speech and not state.is_speaking:
                # Потенциальное начало речи
                if state.speech_start_time is None:
                    state.speech_start_time = time.time()
                    self.logger.debug(
                        "VAD: Potential speech start detected",
                        session_id=session_id,
                        energy=energy,
                        threshold=self.energy_threshold
                    )
                
                # Проверяем достаточно ли чанков с речью
                recent_chunks = state.audio_energy_history[-self.speech_start_chunks:]
                speech_chunks = sum(1 for e in recent_chunks if e > self.energy_threshold)
                
                if speech_chunks >= self.speech_start_chunks:
                    # Подтверждено начало речи!
                    state.is_speaking = True
                    state.silence_start_time = None
                    
                    self.logger.info(
                        "VAD: Speech started",
                        session_id=session_id,
                        energy=energy,
                        detection_latency_ms=int((time.time() - state.speech_start_time) * 1000),
                        time_since_recording_start=f"{time_since_start:.2f}s"
                    )
                    
                    # Уведомляем о начале речи
                    if not state.thinking_sent and self._speech_started_callback:
                        await self._speech_started_callback(session_id)
                        state.thinking_sent = True
                        state.thinking_sent_timestamp = current_time
                    
                    return True
                    
            elif not is_speech and state.is_speaking:
                # Потенциальный конец речи
                if state.silence_start_time is None:
                    state.silence_start_time = time.time()
                
                # Проверяем достаточно ли тишины
                recent_chunks = state.audio_energy_history[-self.silence_chunks:]
                silence_chunks = sum(1 for e in recent_chunks if e <= self.energy_threshold)
                
                if silence_chunks >= self.silence_chunks:
                    # Конец речи
                    state.is_speaking = False
                    state.speech_start_time = None
                    
                    self.logger.info(
                        "VAD: Speech ended",
                        session_id=session_id,
                        silence_duration_ms=int((time.time() - state.silence_start_time) * 1000)
                    )
            
            return False
            
        except Exception as e:
            self.logger.error(
                "Error in VAD processing",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def reset_session(self, session_id: str):
        """Сбросить состояние сессии"""
        async with self._state_lock:
            if session_id in self._session_states:
                # Сохраняем время начала записи если оно есть
                old_recording_start = self._session_states[session_id].recording_start_time
                self._session_states[session_id] = VADState()
                # Восстанавливаем время начала записи для корректной работы
                if old_recording_start:
                    self._session_states[session_id].recording_start_time = old_recording_start
                self.logger.debug(
                    "VAD: Session reset",
                    session_id=session_id
                )
    
    async def cleanup_session(self, session_id: str):
        """Очистить данные сессии"""
        async with self._state_lock:
            if session_id in self._session_states:
                del self._session_states[session_id]
                self.logger.debug(
                    "VAD: Session cleaned up", 
                    session_id=session_id
                )
    
    def get_stats(self, session_id: str) -> dict:
        """Получить статистику для сессии"""
        state = self._session_states.get(session_id)
        if not state:
            return {}
        
        return {
            "is_speaking": state.is_speaking,
            "thinking_sent": state.thinking_sent,
            "energy_history_length": len(state.audio_energy_history),
            "average_energy": np.mean(state.audio_energy_history) if state.audio_energy_history else 0
        }