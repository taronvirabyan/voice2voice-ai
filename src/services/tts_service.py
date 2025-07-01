"""
Sber SaluteSpeech TTS (Text-to-Speech) сервис
КРИТИЧЕСКИ ВАЖНО: Оптимизированная реализация для минимальной латентности
"""

import asyncio
import httpx
import base64
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any, List
import io

from ..core.config import settings
from ..core.models import VoiceResponse
from ..core.exceptions import SaluteSpeechError, TTSError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance


class SaluteSpeechTTS(LoggerMixin):
    """
    Сервис синтеза речи через Sber SaluteSpeech API
    Оптимизирован для минимальной латентности и высокого качества
    """
    
    def __init__(self):
        self.client_id = settings.salute_client_id
        self.client_secret = settings.salute_client_secret
        self.auth_url = settings.salute_auth_url
        self.tts_url = settings.salute_tts_url
        self.voice = settings.salute_voice
        
        # Кеширование токена (используем тот же что и для STT)
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._token_lock = asyncio.Lock()
        
        # HTTP клиент
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Кеш синтезированных фраз для ускорения
        self._synthesis_cache: Dict[str, bytes] = {}
        self._cache_max_size = 100
    
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
    
    async def prewarm(self) -> bool:
        """
        Предварительный прогрев TTS сервиса для ускорения первого синтеза
        
        Returns:
            bool: True если прогрев успешен
        """
        try:
            # Инициализируем HTTP клиент заранее
            if not self._http_client:
                await self._ensure_http_client()
            
            # Получаем токен заранее
            await self._get_access_token()
            
            self.logger.info("SaluteSpeech TTS service prewarmed successfully")
            return True
            
        except Exception as e:
            self.logger.warning(
                "Failed to prewarm TTS service",
                error=str(e)
            )
            return False
    
    async def _get_access_token(self) -> str:
        """
        Получение токена авторизации (переиспользуем логику из STT)
        """
        async with self._token_lock:
            # Проверяем кеш (с запасом в 60 секунд)
            if self._access_token and time.time() < (self._token_expires_at - 60):
                return self._access_token
            
            start_time = time.time()
            
            try:
                await self._ensure_http_client()
                
                auth_data = {
                    "scope": "SALUTE_SPEECH_PERS",
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
                
                self.logger.info(
                    "Requesting SaluteSpeech TTS auth token",
                    **log_request("auth", "SaluteSpeech-TTS")
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
                        "SaluteSpeech TTS auth failed",
                        **log_error(
                            SaluteSpeechError(f"TTS Auth failed: {error_text}",
                                            status_code=response.status_code),
                            "auth",
                            "SaluteSpeech-TTS"
                        )
                    )
                    raise SaluteSpeechError(
                        f"TTS Authentication failed: {error_text}",
                        status_code=response.status_code,
                        response_body=error_text
                    )
                
                auth_response = response.json()
                
                if "access_token" not in auth_response:
                    raise SaluteSpeechError("No access_token in TTS response")
                
                self._access_token = auth_response["access_token"]
                expires_in = auth_response.get("expires_in", 3600)
                self._token_expires_at = time.time() + expires_in
                
                self.logger.info(
                    "SaluteSpeech TTS auth successful",
                    **log_performance("auth", duration_ms, "SaluteSpeech-TTS"),
                    expires_in=expires_in
                )
                
                return self._access_token
                
            except httpx.RequestError as e:
                self.logger.error(
                    "SaluteSpeech TTS auth network error",
                    **log_error(e, "auth", "SaluteSpeech-TTS")
                )
                raise SaluteSpeechError(f"TTS Network error during auth: {str(e)}")
            except Exception as e:
                self.logger.error(
                    "SaluteSpeech TTS auth unexpected error",
                    **log_error(e, "auth", "SaluteSpeech-TTS")
                )
                raise SaluteSpeechError(f"TTS Unexpected error during auth: {str(e)}")
    
    def _validate_text(self, text: str) -> str:
        """
        Валидация и нормализация текста для синтеза
        """
        if not text or not text.strip():
            raise TTSError("Empty text for synthesis")
        
        # Нормализуем текст
        text = text.strip()
        
        # Проверяем длину (ограничение API)
        if len(text) > 1000:
            self.logger.warning(
                "Text too long for TTS, truncating",
                original_length=len(text),
                max_length=1000
            )
            text = text[:997] + "..."
        
        # Базовая очистка
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Убираем множественные пробелы
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        return text
    
    def _get_cache_key(self, text: str, voice: str) -> str:
        """Создание ключа для кеша"""
        return f"{voice}:{hash(text)}"
    
    def _add_to_cache(self, key: str, audio_data: bytes) -> None:
        """Добавление в кеш с ограничением размера"""
        if len(self._synthesis_cache) >= self._cache_max_size:
            # Удаляем самый старый элемент (FIFO)
            oldest_key = next(iter(self._synthesis_cache))
            del self._synthesis_cache[oldest_key]
        
        self._synthesis_cache[key] = audio_data
    
    async def synthesize_text(
        self,
        text: str,
        voice: Optional[str] = None,
        session_id: Optional[str] = None,
        use_cache: bool = True
    ) -> bytes:
        """
        Синтез речи из текста
        
        Args:
            text: Текст для синтеза
            voice: Голос (по умолчанию из настроек)
            session_id: ID сессии для логирования
            use_cache: Использовать кеш
            
        Returns:
            Аудио данные в формате PCM
        """
        # Валидация текста
        text = self._validate_text(text)
        voice = voice or self.voice
        
        # Проверяем кеш
        cache_key = self._get_cache_key(text, voice)
        if use_cache and cache_key in self._synthesis_cache:
            self.logger.debug(
                "TTS cache hit",
                session_id=session_id,
                text_preview=text[:30] + "..." if len(text) > 30 else text
            )
            return self._synthesis_cache[cache_key]
        
        start_time = time.time()
        
        try:
            await self._ensure_http_client()
            
            # Получаем токен
            token = await self._get_access_token()
            
            # Формируем запрос
            request_data = {
                "text": text,
                "voice": voice,
                "audio_encoding": "PCM_S16LE",
                "sample_rate": 24000,  # Высокое качество
                "config": {
                    "speed": 1.0,  # Нормальная скорость
                    "pitch": 0.0,  # Нормальная высота
                    "volume": 1.0  # Нормальная громкость
                }
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "RqUID": str(uuid.uuid4())
            }
            
            self.logger.debug(
                "Sending TTS request",
                **log_request("synthesize", "SaluteSpeech-TTS", session_id),
                text_length=len(text),
                voice=voice
            )
            
            response = await self._http_client.post(
                self.tts_url,
                headers=headers,
                json=request_data
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                error_text = response.text
                self.logger.error(
                    "TTS request failed",
                    **log_error(
                        SaluteSpeechError(f"TTS failed: {error_text}",
                                        status_code=response.status_code),
                        "synthesize",
                        "SaluteSpeech-TTS",
                        session_id
                    )
                )
                
                # Специальная обработка 401 (токен протух)
                if response.status_code == 401:
                    self._access_token = None
                    self._token_expires_at = 0
                
                raise TTSError(f"Synthesis failed: {error_text}")
            
            # Парсим результат
            result = response.json()
            
            if "audio_data" not in result:
                raise TTSError("No audio_data in TTS response")
            
            # Декодируем аудио
            try:
                audio_data = base64.b64decode(result["audio_data"])
            except Exception as e:
                raise TTSError(f"Failed to decode audio data: {str(e)}")
            
            if not audio_data:
                raise TTSError("Empty audio data received")
            
            # Добавляем в кеш
            if use_cache:
                self._add_to_cache(cache_key, audio_data)
            
            self.logger.info(
                "TTS synthesis successful",
                **log_performance("synthesize", duration_ms, "SaluteSpeech-TTS", session_id),
                text_length=len(text),
                audio_size=len(audio_data),
                cached=False
            )
            
            return audio_data
            
        except httpx.RequestError as e:
            self.logger.error(
                "TTS network error",
                **log_error(e, "synthesize", "SaluteSpeech-TTS", session_id)
            )
            raise TTSError(f"Network error during synthesis: {str(e)}")
        except Exception as e:
            if isinstance(e, TTSError):
                raise
            self.logger.error(
                "TTS unexpected error",
                **log_error(e, "synthesize", "SaluteSpeech-TTS", session_id)
            )
            raise TTSError(f"Unexpected error during synthesis: {str(e)}")
    
    async def synthesize_stream(
        self,
        text: str,
        session_id: str,
        chunk_size: int = 4096,
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming синтез речи
        Возвращает аудио по частям для уменьшения latency
        
        Args:
            text: Текст для синтеза
            session_id: ID сессии
            chunk_size: Размер чанка для streaming
            voice: Голос
            
        Yields:
            bytes: Чанки аудио данных
        """
        try:
            # Синтезируем полное аудио
            audio_data = await self.synthesize_text(text, voice, session_id)
            
            # Отдаем по частям с небольшими задержками для плавности
            total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
            
            self.logger.debug(
                "Starting TTS streaming",
                session_id=session_id,
                total_size=len(audio_data),
                chunk_size=chunk_size,
                total_chunks=total_chunks
            )
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk
                
                # Небольшая задержка для контроля скорости потока
                await asyncio.sleep(0.01)
            
            self.logger.debug(
                "TTS streaming completed",
                session_id=session_id,
                chunks_sent=total_chunks
            )
            
        except Exception as e:
            self.logger.error(
                "TTS streaming error",
                **log_error(e, "stream_synthesize", "SaluteSpeech-TTS", session_id)
            )
            raise
    
    async def synthesize_batch(
        self,
        texts: List[str],
        session_id: str,
        voice: Optional[str] = None
    ) -> List[bytes]:
        """
        Пакетный синтез нескольких текстов
        Оптимизирован для предгенерации ответов
        
        Args:
            texts: Список текстов для синтеза
            session_id: ID сессии
            voice: Голос
            
        Returns:
            List[bytes]: Список аудио данных
        """
        self.logger.info(
            "Starting batch TTS synthesis",
            session_id=session_id,
            batch_size=len(texts)
        )
        
        results = []
        
        # Синтезируем параллельно для ускорения
        tasks = [
            self.synthesize_text(text, voice, session_id)
            for text in texts
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем результаты
            audio_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Batch TTS item failed",
                        session_id=session_id,
                        item_index=i,
                        error=str(result)
                    )
                    # Добавляем пустые данные вместо ошибки
                    audio_results.append(b"")
                else:
                    audio_results.append(result)
            
            self.logger.info(
                "Batch TTS synthesis completed",
                session_id=session_id,
                successful_items=sum(1 for r in results if not isinstance(r, Exception)),
                failed_items=sum(1 for r in results if isinstance(r, Exception))
            )
            
            return audio_results
            
        except Exception as e:
            self.logger.error(
                "Batch TTS synthesis error",
                **log_error(e, "batch_synthesize", "SaluteSpeech-TTS", session_id)
            )
            raise
    
    async def health_check(self) -> bool:
        """
        Проверка доступности TTS сервиса
        """
        try:
            # Проверяем через синтез тестовой фразы
            test_audio = await self.synthesize_text(
                "Проверка работоспособности",
                use_cache=False
            )
            return len(test_audio) > 0
        except Exception as e:
            self.logger.error(
                "TTS health check failed",
                **log_error(e, "health_check", "SaluteSpeech-TTS")
            )
            return False
    
    def clear_cache(self) -> None:
        """Очистка кеша синтеза"""
        cache_size = len(self._synthesis_cache)
        self._synthesis_cache.clear()
        self.logger.info(
            "TTS cache cleared",
            cleared_items=cache_size
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кеша"""
        return {
            "cache_size": len(self._synthesis_cache),
            "cache_max_size": self._cache_max_size,
            "cache_usage_percent": (len(self._synthesis_cache) / self._cache_max_size) * 100
        }