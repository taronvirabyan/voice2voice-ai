"""
Mock TTS Service для демонстрации полной функциональности
КРИТИЧЕСКИ ВАЖНО: Обеспечивает 100% uptime для тестирования
"""

import asyncio
import time
import io
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
import wave

from ..core.config import settings
from ..core.exceptions import TTSError
from ..core.logging import LoggerMixin, log_performance


class MockTTSService(LoggerMixin):
    """
    Mock TTS Service для демонстрации функциональности
    Генерирует синтетические аудио данные
    """
    
    def __init__(self):
        """Инициализация Mock TTS сервиса"""
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2  # 16-bit
        
        self.logger.info(
            "MockTTS service initializing",
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width
        )
    
    async def health_check(self) -> bool:
        """Проверка здоровья Mock TTS сервиса (всегда успешна)"""
        try:
            # Генерируем тестовый аудио сигнал
            test_audio = await self._generate_audio("Test", duration=0.1)
            
            success = test_audio is not None and len(test_audio) > 0
            
            self.logger.info(
                "MockTTS health check completed",
                success=success,
                test_audio_size=len(test_audio) if test_audio else 0
            )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "MockTTS health check failed",
                error=str(e)
            )
            return False
    
    async def synthesize_stream(
        self,
        text: str,
        session_id: str,
        voice_id: Optional[str] = None,
        chunk_size: int = 1024
    ) -> AsyncGenerator[bytes, None]:
        """
        Потоковый синтез речи (Mock)
        
        Args:
            text: Текст для синтеза
            session_id: ID сессии
            voice_id: ID голоса (игнорируется в mock)
            chunk_size: Размер чанка для streaming
            
        Yields:
            bytes: Чанки аудио данных
        """
        self.logger.info(
            "Starting MockTTS synthesis",
            session_id=session_id,
            text_length=len(text),
            text_preview=text[:50] + "..." if len(text) > 50 else text
        )
        
        try:
            # Генерируем аудио на основе длины текста
            duration = max(1.0, len(text) * 0.05)  # ~0.05 сек на символ
            audio_data = await self._generate_audio(text, duration)
            
            if audio_data is None or len(audio_data) == 0:
                self.logger.error(
                    "No audio data generated",
                    session_id=session_id,
                    text_length=len(text)
                )
                raise TTSError("No audio data generated")
            
            # Отправляем аудио чанками для streaming
            total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                self.logger.debug(
                    "Streaming audio chunk",
                    session_id=session_id,
                    chunk_number=i // chunk_size + 1,
                    total_chunks=total_chunks,
                    chunk_size=len(chunk)
                )
                
                yield chunk
                
                # Небольшая пауза для плавного streaming
                await asyncio.sleep(0.02)
            
            self.logger.info(
                "✅ MockTTS synthesis completed",
                session_id=session_id,
                total_audio_size=len(audio_data),
                total_chunks=total_chunks,
                duration_seconds=duration
            )
            
        except Exception as e:
            self.logger.error(
                "MockTTS synthesis failed",
                error=str(e),
                session_id=session_id,
                text_length=len(text)
            )
            raise TTSError(f"Mock TTS synthesis failed: {str(e)}")
    
    async def _generate_audio(self, text: str, duration: float) -> bytes:
        """
        Генерация синтетического аудио сигнала
        
        Args:
            text: Текст (влияет на частотные характеристики)
            duration: Длительность в секундах
            
        Returns:
            bytes: WAV аудио данные
        """
        try:
            start_time = time.time()
            
            # Вычисляем параметры сигнала на основе текста
            text_hash = hash(text) % 1000
            base_freq = 200 + (text_hash % 300)  # Частота 200-500 Hz
            
            # Генерируем сэмплы
            num_samples = int(self.sample_rate * duration)
            t = np.linspace(0, duration, num_samples, dtype=np.float32)
            
            # Создаем сложный сигнал (имитация речи)
            signal = np.zeros_like(t)
            
            # Основная частота
            signal += 0.3 * np.sin(2 * np.pi * base_freq * t)
            
            # Гармоники
            signal += 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
            signal += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Модуляция (имитация интонации)
            modulation = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
            signal *= modulation
            
            # Добавляем небольшой шум (имитация естественности)
            noise = 0.05 * np.random.random(num_samples) - 0.025
            signal += noise
            
            # Нормализация
            signal = np.clip(signal, -1.0, 1.0)
            
            # Конвертируем в 16-bit PCM
            audio_int16 = (signal * 32767).astype(np.int16)
            
            # Создаем WAV файл в памяти
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()
            
            generation_time = time.time() - start_time
            
            self.logger.info(
                "Audio generation completed",
                **log_performance("mock_tts_generation", generation_time * 1000, "MockTTS"),
                text_length=len(text),
                audio_size=len(wav_data),
                duration_seconds=duration,
                base_frequency=base_freq
            )
            
            return wav_data
            
        except Exception as e:
            self.logger.error(
                "Audio generation failed",
                error=str(e),
                text_length=len(text),
                duration=duration
            )
            raise TTSError(f"Audio generation failed: {str(e)}")
    
    async def get_voice_info(self) -> Dict[str, Any]:
        """Информация о доступных голосах Mock TTS"""
        return {
            "service": "MockTTS",
            "voices": [
                {
                    "id": "mock_voice_1",
                    "name": "Mock Voice 1",
                    "language": "ru",
                    "description": "Synthetic voice for testing"
                }
            ],
            "features": [
                "Real-time synthesis",
                "Text-based frequency generation",
                "WAV format output",
                "Streaming support"
            ]
        }
    
    async def shutdown(self):
        """Корректное завершение работы сервиса"""
        self.logger.info("MockTTS service shutting down")