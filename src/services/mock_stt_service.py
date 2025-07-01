"""
Mock STT Service для демонстрации полной voice2voice функциональности
КРИТИЧЕСКИ ВАЖНО: Обеспечивает STT функциональность без больших зависимостей
"""

import asyncio
import time
import random
import wave
import io
from typing import AsyncGenerator, Optional, Dict, Any
import base64

from ..core.config import settings
from ..core.exceptions import AudioProcessingError, STTError
from ..core.logging import LoggerMixin, log_performance


class MockSTTService(LoggerMixin):
    """
    Mock STT Service для демонстрации функциональности
    Имитирует распознавание речи для тестирования
    """
    
    def __init__(self):
        """Инициализация Mock STT сервиса"""
        self.model_name = "MockSTT-v1.0"
        self.language = "ru"
        self.sample_rate = 16000
        
        # Предустановленные фразы для различных аудио паттернов
        self.recognition_patterns = [
            "Привет, как дела?",
            "Расскажи мне про собак",
            "У меня проблемы с деньгами",
            "Мне нужна работа",
            "Что ты можешь мне предложить?",
            "Это тестовое сообщение",
            "Проверка системы распознавания речи",
            "Здравствуйте, я хочу поговорить",
        ]
        
        self.logger.info(
            "MockSTT service initializing",
            model_name=self.model_name,
            language=self.language,
            sample_rate=self.sample_rate
        )
    
    async def health_check(self) -> bool:
        """Проверка здоровья Mock STT сервиса (всегда успешна)"""
        try:
            # Тестируем распознавание
            test_audio = self._generate_test_audio()
            test_result = await self._process_audio(test_audio)
            
            success = test_result is not None and len(test_result.strip()) > 0
            
            self.logger.info(
                "MockSTT health check completed",
                success=success,
                test_result_length=len(test_result) if test_result else 0
            )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "MockSTT health check failed",
                error=str(e)
            )
            return False
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        session_id: str,
        language: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Потоковое распознавание речи
        
        Args:
            audio_stream: Поток аудио данных
            session_id: ID сессии
            language: Язык распознавания
            
        Yields:
            Dict с результатами распознавания
        """
        self.logger.info(
            "Starting MockSTT transcription",
            session_id=session_id,
            language=language or self.language
        )
        
        try:
            audio_buffer = bytearray()
            chunk_count = 0
            
            async for chunk in audio_stream:
                audio_buffer.extend(chunk)
                chunk_count += 1
                
                # Имитируем обработку после накопления достаточного количества данных
                if len(audio_buffer) >= self.sample_rate * 2:  # 2 секунды аудио
                    # Процессим аудио
                    transcription = await self._process_audio(bytes(audio_buffer))
                    
                    if transcription:
                        yield {
                            "type": "transcription",
                            "text": transcription,
                            "is_final": False,
                            "confidence": random.uniform(0.85, 0.98),
                            "timestamp": time.time()
                        }
                    
                    # Очищаем буфер
                    audio_buffer = bytearray()
                    
                # Имитируем промежуточные результаты
                if chunk_count % 5 == 0 and len(audio_buffer) > 0:
                    partial_text = "..." 
                    yield {
                        "type": "partial",
                        "text": partial_text,
                        "is_final": False,
                        "timestamp": time.time()
                    }
            
            # Обрабатываем оставшиеся данные
            if len(audio_buffer) > 0:
                final_transcription = await self._process_audio(bytes(audio_buffer))
                
                if final_transcription:
                    yield {
                        "type": "transcription",
                        "text": final_transcription,
                        "is_final": True,
                        "confidence": random.uniform(0.90, 0.99),
                        "timestamp": time.time()
                    }
            
            self.logger.info(
                "✅ MockSTT transcription completed",
                session_id=session_id,
                total_chunks=chunk_count
            )
            
        except Exception as e:
            self.logger.error(
                "MockSTT transcription failed",
                error=str(e),
                session_id=session_id
            )
            raise STTError(f"Mock STT transcription failed: {str(e)}")
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        session_id: str,
        language: Optional[str] = None
    ) -> str:
        """
        Распознавание полного аудио файла
        
        Args:
            audio_data: Аудио данные
            session_id: ID сессии  
            language: Язык распознавания
            
        Returns:
            str: Распознанный текст
        """
        self.logger.info(
            "MockSTT transcribing audio",
            session_id=session_id,
            audio_size=len(audio_data),
            language=language or self.language
        )
        
        try:
            transcription = await self._process_audio(audio_data)
            
            self.logger.info(
                "✅ MockSTT transcription completed",
                session_id=session_id,
                text_length=len(transcription),
                text_preview=transcription[:50] + "..." if len(transcription) > 50 else transcription
            )
            
            return transcription
            
        except Exception as e:
            self.logger.error(
                "MockSTT transcription failed",
                error=str(e),
                session_id=session_id,
                audio_size=len(audio_data)
            )
            raise STTError(f"Mock STT transcription failed: {str(e)}")
    
    async def _process_audio(self, audio_data: bytes) -> str:
        """
        Внутренняя обработка аудио данных
        """
        start_time = time.time()
        
        # Симулируем время обработки
        await asyncio.sleep(random.uniform(0.2, 0.8))
        
        # Анализируем "характеристики" аудио для выбора фразы
        audio_hash = hash(audio_data) % len(self.recognition_patterns)
        
        # Выбираем фразу на основе характеристик
        base_text = self.recognition_patterns[audio_hash]
        
        # Добавляем вариативность
        if random.random() < 0.2:  # 20% шанс модификации
            modifications = [
                " Пожалуйста.",
                " Спасибо.",
                " Это важно для меня.",
                " Можете помочь?",
            ]
            base_text += random.choice(modifications)
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            "Audio processing completed",
            **log_performance("mock_stt_processing", processing_time * 1000, "MockSTT"),
            audio_size=len(audio_data),
            text_length=len(base_text)
        )
        
        return base_text
    
    def _generate_test_audio(self) -> bytes:
        """Генерация тестового аудио для health check"""
        # Создаем минимальный WAV файл
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            # Пишем 1 секунду тишины
            wav_file.writeframes(b'\x00\x00' * self.sample_rate)
        
        return wav_buffer.getvalue()
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Информация о Mock STT сервисе"""
        return {
            "service": "MockSTT",
            "model": self.model_name,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "capabilities": [
                "Speech recognition",
                "Russian language",
                "Streaming transcription",
                "Batch transcription",
                "Real-time processing"
            ],
            "features": [
                "100% uptime",
                "No external dependencies",
                "Simulated recognition",
                "Performance metrics",
                "Session support"
            ],
            "recognition_patterns": len(self.recognition_patterns)
        }
    
    async def shutdown(self):
        """Корректное завершение работы сервиса"""
        self.logger.info("MockSTT service shutting down")