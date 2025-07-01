"""
Sber SaluteSpeech STT (Speech-to-Text) сервис
КРИТИЧЕСКИ ВАЖНО: Проверенная реализация с обработкой всех ошибок
"""

import asyncio
import httpx
import base64
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np

from ..core.config import settings
from ..core.models import TranscriptSegment, MessageRole
from ..core.exceptions import SaluteSpeechError, AudioProcessingError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance


class SaluteSpeechSTT(LoggerMixin):
    """
    Сервис распознавания речи через Sber SaluteSpeech API
    Реализует streaming STT с обработкой ошибок и retry логикой
    """
    
    def __init__(self):
        self.client_id = settings.salute_client_id
        self.client_secret = settings.salute_client_secret
        self.auth_url = settings.salute_auth_url
        self.stt_url = settings.salute_stt_url
        
        # Кеширование токена
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._token_lock = asyncio.Lock()
        
        # HTTP клиент с правильными настройками
        self._http_client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_http_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def _ensure_http_client(self) -> None:
        """Создание HTTP клиента если не существует"""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=settings.request_timeout,
                    write=10.0,
                    pool=30.0
                ),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={
                    "User-Agent": "Voice2Voice-AI/1.0",
                    "Accept": "application/json",
                }
            )
    
    async def _get_access_token(self) -> str:
        """
        Получение токена авторизации с кешированием
        ВАЖНО: Токен кешируется для оптимизации
        """
        async with self._token_lock:
            # Проверяем кеш (с запасом в 60 секунд)
            if self._access_token and time.time() < (self._token_expires_at - 60):
                return self._access_token
            
            start_time = time.time()
            
            try:
                await self._ensure_http_client()
                
                # Формируем запрос авторизации
                auth_data = {
                    "scope": "SALUTE_SPEECH_PERS",
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
                
                self.logger.info(
                    "Requesting SaluteSpeech auth token",
                    **log_request("auth", "SaluteSpeech")
                )
                
                response = await self._http_client.post(
                    self.auth_url,
                    headers={"RqUID": str(uuid.uuid4())},
                    data=auth_data
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status_code != 200:
                    error_text = response.text
                    self.logger.error(
                        "SaluteSpeech auth failed",
                        **log_error(
                            SaluteSpeechError(f"Auth failed: {error_text}", 
                                            status_code=response.status_code),
                            "auth",
                            "SaluteSpeech"
                        )
                    )
                    raise SaluteSpeechError(
                        f"Authentication failed: {error_text}",
                        status_code=response.status_code,
                        response_body=error_text
                    )
                
                # Парсим ответ
                auth_response = response.json()
                
                if "access_token" not in auth_response:
                    raise SaluteSpeechError("No access_token in response")
                
                self._access_token = auth_response["access_token"]
                expires_in = auth_response.get("expires_in", 3600)
                self._token_expires_at = time.time() + expires_in
                
                self.logger.info(
                    "SaluteSpeech auth successful",
                    **log_performance("auth", duration_ms, "SaluteSpeech"),
                    expires_in=expires_in
                )
                
                return self._access_token
                
            except httpx.RequestError as e:
                self.logger.error(
                    "SaluteSpeech auth network error",
                    **log_error(e, "auth", "SaluteSpeech")
                )
                raise SaluteSpeechError(f"Network error during auth: {str(e)}")
            except Exception as e:
                self.logger.error(
                    "SaluteSpeech auth unexpected error",
                    **log_error(e, "auth", "SaluteSpeech")
                )
                raise SaluteSpeechError(f"Unexpected error during auth: {str(e)}")
    
    def _prepare_audio_data(self, audio_bytes: bytes) -> str:
        """
        Подготовка аудио данных для API
        Конвертация в base64 с валидацией
        """
        if not audio_bytes:
            raise AudioProcessingError("Empty audio data")
        
        if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB лимит
            raise AudioProcessingError("Audio data too large")
        
        try:
            return base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            raise AudioProcessingError(f"Failed to encode audio: {str(e)}")
    
    async def transcribe_audio_chunk(
        self, 
        audio_data: bytes,
        language: str = "ru-RU",
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Транскрибация одного чанка аудио
        
        Args:
            audio_data: Аудио данные в формате PCM 16bit
            language: Язык распознавания
            session_id: ID сессии для логирования
            
        Returns:
            Распознанный текст или None если не распознано
        """
        if not audio_data or len(audio_data) < 1000:  # Минимум данных
            return None
            
        start_time = time.time()
        
        try:
            await self._ensure_http_client()
            
            # Получаем токен авторизации
            token = await self._get_access_token()
            
            # Подготавливаем аудио
            audio_base64 = self._prepare_audio_data(audio_data)
            
            # Формируем запрос
            request_data = {
                "audio": {
                    "audio_encoding": "PCM_S16LE",
                    "sample_rate": settings.sample_rate,
                    "audio_data": audio_base64
                },
                "language": language,
                "config": {
                    "enable_partial_results": True,
                    "enable_automatic_punctuation": True,
                    "speech_contexts": [{
                        "phrases": ["собака", "кошка", "деньги", "дети", "ошейник", "MLM", "курсы"]
                    }]
                }
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "RqUID": str(uuid.uuid4())
            }
            
            self.logger.debug(
                "Sending STT request",
                **log_request("transcribe", "SaluteSpeech", session_id),
                audio_size=len(audio_data),
                language=language
            )
            
            response = await self._http_client.post(
                self.stt_url,
                headers=headers,
                json=request_data
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                error_text = response.text
                self.logger.error(
                    "STT request failed",
                    **log_error(
                        SaluteSpeechError(f"STT failed: {error_text}",
                                        status_code=response.status_code),
                        "transcribe",
                        "SaluteSpeech",
                        session_id
                    )
                )
                
                # Специальная обработка 401 (токен протух)
                if response.status_code == 401:
                    self._access_token = None
                    self._token_expires_at = 0
                
                return None
            
            # Парсим результат
            result = response.json()
            
            if "text" in result and result["text"]:
                text = result["text"].strip()
                confidence = result.get("confidence", 0.0)
                
                self.logger.info(
                    "STT transcription successful",
                    **log_performance("transcribe", duration_ms, "SaluteSpeech", session_id),
                    text_length=len(text),
                    confidence=confidence
                )
                
                return text
            else:
                self.logger.debug(
                    "STT no text recognized",
                    **log_performance("transcribe", duration_ms, "SaluteSpeech", session_id)
                )
                return None
                
        except httpx.RequestError as e:
            self.logger.error(
                "STT network error",
                **log_error(e, "transcribe", "SaluteSpeech", session_id)
            )
            return None
        except Exception as e:
            self.logger.error(
                "STT unexpected error",
                **log_error(e, "transcribe", "SaluteSpeech", session_id)
            )
            return None
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        session_id: str,
        chunk_duration: float = 1.0
    ) -> AsyncGenerator[TranscriptSegment, None]:
        """
        Streaming транскрипция аудио потока
        
        Args:
            audio_stream: Поток аудио данных
            session_id: ID сессии
            chunk_duration: Длительность чанка в секундах
            
        Yields:
            TranscriptSegment: Сегменты распознанного текста
        """
        self.logger.info(
            "Starting STT streaming",
            session_id=session_id,
            chunk_duration=chunk_duration
        )
        
        buffer = bytearray()
        chunk_size = int(settings.sample_rate * chunk_duration * 2)  # 16-bit = 2 bytes
        sequence_number = 0
        
        try:
            async for audio_chunk in audio_stream:
                if not audio_chunk:
                    continue
                    
                buffer.extend(audio_chunk)
                
                # Когда накопили достаточно данных
                while len(buffer) >= chunk_size:
                    # Извлекаем чанк
                    chunk_data = bytes(buffer[:chunk_size])
                    buffer = buffer[chunk_size:]
                    
                    # Транскрибируем
                    text = await self.transcribe_audio_chunk(
                        chunk_data, 
                        session_id=session_id
                    )
                    
                    if text:
                        segment = TranscriptSegment(
                            text=text,
                            speaker=MessageRole.USER,
                            confidence=None,  # SaluteSpeech не всегда возвращает confidence
                            timestamp=time.time()
                        )
                        
                        self.logger.debug(
                            "STT segment recognized",
                            session_id=session_id,
                            sequence=sequence_number,
                            text=text[:50] + "..." if len(text) > 50 else text
                        )
                        
                        yield segment
                    
                    sequence_number += 1
                    
        except Exception as e:
            self.logger.error(
                "STT streaming error",
                **log_error(e, "stream_transcribe", "SaluteSpeech", session_id)
            )
            raise
        
        finally:
            # Обрабатываем оставшиеся данные в буфере
            if len(buffer) > 1000:  # Минимум для обработки
                text = await self.transcribe_audio_chunk(
                    bytes(buffer),
                    session_id=session_id
                )
                
                if text:
                    segment = TranscriptSegment(
                        text=text,
                        speaker=MessageRole.USER,
                        timestamp=time.time()
                    )
                    
                    self.logger.info(
                        "STT final segment",
                        session_id=session_id,
                        text=text[:50] + "..." if len(text) > 50 else text
                    )
                    
                    yield segment
            
            self.logger.info(
                "STT streaming completed",
                session_id=session_id,
                total_chunks=sequence_number
            )
    
    async def health_check(self) -> bool:
        """
        Проверка доступности сервиса
        Возвращает True если сервис работает
        """
        try:
            token = await self._get_access_token()
            return bool(token)
        except Exception as e:
            self.logger.error(
                "STT health check failed",
                **log_error(e, "health_check", "SaluteSpeech")
            )
            return False