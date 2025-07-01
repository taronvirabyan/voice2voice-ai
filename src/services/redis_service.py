"""
Production-Ready Redis Service для Voice2Voice системы
КРИТИЧЕСКИ ВАЖНО: Обеспечивает надежное межсервисное взаимодействие
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, AsyncGenerator, Any
import redis.asyncio as redis
from redis.asyncio import Redis

from ..core.config import settings
from ..core.models import Session, TranscriptSegment, PromptUpdate, SessionMetrics, SystemHealth
from ..core.exceptions import RedisError, SessionError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance


class RedisService(LoggerMixin):
    """
    Production-Ready Redis сервис для управления сессиями и коммуникации
    Обеспечивает надежное хранение состояния и pub/sub коммуникацию
    """
    
    def __init__(self):
        self.redis_url = settings.redis_url
        self.redis_db = settings.redis_db
        self.redis_timeout = settings.redis_timeout
        
        # Connection pool для оптимизации
        self._pool: Optional[redis.ConnectionPool] = None
        self._redis: Optional[Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        
        # Настройки префиксов для ключей
        self.key_prefixes = {
            "session": "session:",
            "transcript": "transcript:",
            "prompt_update": "prompt_update:",
            "metrics": "metrics:",
            "health": "health:",
            "cache": "cache:"
        }
        
        # Lock для управления соединениями
        self._connection_lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Установка соединения с Redis"""
        async with self._connection_lock:
            if self._redis and await self._redis.ping():
                return  # Уже подключены
            
            try:
                start_time = time.time()
                
                # Создаем connection pool
                self._pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    db=self.redis_db,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    socket_connect_timeout=self.redis_timeout,
                    decode_responses=True
                )
                
                # Создаем Redis клиент
                self._redis = Redis(connection_pool=self._pool)
                
                # Проверяем соединение
                await self._redis.ping()
                
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    "Redis connection established",
                    **log_performance("connect", duration_ms, "Redis"),
                    redis_url=self.redis_url,
                    db=self.redis_db
                )
                
            except Exception as e:
                self.logger.error(
                    "Redis connection failed",
                    **log_error(e, "connect", "Redis")
                )
                raise RedisError(f"Failed to connect to Redis: {str(e)}")
    
    async def disconnect(self) -> None:
        """Закрытие соединения с Redis"""
        async with self._connection_lock:
            try:
                if self._pubsub:
                    await self._pubsub.close()
                    self._pubsub = None
                
                if self._redis:
                    await self._redis.close()
                    self._redis = None
                
                if self._pool:
                    await self._pool.disconnect()
                    self._pool = None
                
                self.logger.info("Redis connection closed")
                
            except Exception as e:
                self.logger.error(
                    "Error closing Redis connection",
                    **log_error(e, "disconnect", "Redis")
                )
    
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Генерация ключа Redis с префиксом"""
        return f"{self.key_prefixes[prefix]}{identifier}"
    
    async def _ensure_connected(self) -> None:
        """Проверка соединения и переподключение если нужно"""
        if not self._redis:
            await self.connect()
            return
        
        try:
            await self._redis.ping()
        except Exception:
            self.logger.warning("Redis connection lost, reconnecting...")
            await self.connect()
    
    # ===== УПРАВЛЕНИЕ СЕССИЯМИ =====
    
    async def create_session(self, session_id: str, initial_data: Optional[Dict] = None) -> Session:
        """
        Создание новой сессии
        
        Args:
            session_id: Уникальный ID сессии
            initial_data: Начальные данные сессии
            
        Returns:
            Session: Созданная сессия
        """
        await self._ensure_connected()
        
        try:
            # Создаем объект сессии
            session_data = initial_data or {}
            session = Session(id=session_id, **session_data)
            
            # Сохраняем в Redis с TTL
            session_key = self._get_key("session", session_id)
            
            await self._redis.setex(
                session_key,
                settings.session_timeout,
                session.json()
            )
            
            self.logger.info(
                "Session created",
                session_id=session_id,
                ttl=settings.session_timeout
            )
            
            return session
            
        except Exception as e:
            self.logger.error(
                "Failed to create session",
                **log_error(e, "create_session", "Redis"),
                session_id=session_id
            )
            raise RedisError(f"Failed to create session: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Получение сессии по ID
        
        Args:
            session_id: ID сессии
            
        Returns:
            Session или None если не найдена
        """
        await self._ensure_connected()
        
        try:
            session_key = self._get_key("session", session_id)
            data = await self._redis.get(session_key)
            
            if not data:
                return None
            
            session = Session.parse_raw(data)
            
            self.logger.debug(
                "Session retrieved",
                session_id=session_id,
                state=session.state
            )
            
            return session
            
        except Exception as e:
            self.logger.error(
                "Failed to get session",
                **log_error(e, "get_session", "Redis"),
                session_id=session_id
            )
            return None
    
    async def update_session(self, session: Session) -> bool:
        """
        Обновление сессии
        
        Args:
            session: Объект сессии для обновления
            
        Returns:
            bool: Успешность операции
        """
        await self._ensure_connected()
        
        try:
            session_key = self._get_key("session", session.id)
            
            # Обновляем timestamp
            session.updated_at = session.updated_at.__class__.now()
            
            # Сохраняем с продлением TTL
            result = await self._redis.setex(
                session_key,
                settings.session_timeout,
                session.json()
            )
            
            self.logger.debug(
                "Session updated",
                session_id=session.id,
                state=session.state
            )
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(
                "Failed to update session",
                **log_error(e, "update_session", "Redis"),
                session_id=session.id
            )
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Удаление сессии и связанных данных
        
        Args:
            session_id: ID сессии
            
        Returns:
            bool: Успешность операции
        """
        await self._ensure_connected()
        
        try:
            # Удаляем основную сессию
            session_key = self._get_key("session", session_id)
            
            # Удаляем связанные streams
            transcript_key = self._get_key("transcript", session_id)
            
            # Batch операция для атомарности
            pipe = self._redis.pipeline()
            pipe.delete(session_key)
            pipe.delete(transcript_key)
            results = await pipe.execute()
            
            self.logger.info(
                "Session deleted",
                session_id=session_id,
                deleted_keys=sum(results)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to delete session",
                **log_error(e, "delete_session", "Redis"),
                session_id=session_id
            )
            return False
    
    # ===== УПРАВЛЕНИЕ ТРАНСКРИПТАМИ =====
    
    async def add_transcript_segment(
        self, 
        session_id: str, 
        segment: TranscriptSegment
    ) -> bool:
        """
        Добавление сегмента транскрипта
        
        Args:
            session_id: ID сессии
            segment: Сегмент транскрипта
            
        Returns:
            bool: Успешность операции
        """
        await self._ensure_connected()
        
        try:
            transcript_key = self._get_key("transcript", session_id)
            
            # Добавляем в Redis Stream
            stream_data = {
                "text": segment.text,
                "speaker": segment.speaker.value,
                "timestamp": str(segment.timestamp),
                "confidence": str(segment.confidence) if segment.confidence else "null"
            }
            
            # Добавляем с автоматическим ID
            message_id = await self._redis.xadd(transcript_key, stream_data)
            
            # Ограничиваем размер stream (последние 100 сообщений)
            await self._redis.xtrim(transcript_key, maxlen=100, approximate=True)
            
            self.logger.debug(
                "Transcript segment added",
                session_id=session_id,
                speaker=segment.speaker.value,
                text_length=len(segment.text),
                message_id=message_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to add transcript segment",
                **log_error(e, "add_transcript", "Redis"),
                session_id=session_id
            )
            return False
    
    async def get_transcript_history(
        self, 
        session_id: str, 
        count: int = 20
    ) -> List[TranscriptSegment]:
        """
        Получение истории транскрипта
        
        Args:
            session_id: ID сессии
            count: Количество последних сегментов
            
        Returns:
            List[TranscriptSegment]: История транскрипта
        """
        await self._ensure_connected()
        
        try:
            transcript_key = self._get_key("transcript", session_id)
            
            # Получаем последние сообщения из stream
            messages = await self._redis.xrevrange(transcript_key, count=count)
            
            segments = []
            for message_id, fields in reversed(messages):  # Восстанавливаем порядок
                try:
                    segment = TranscriptSegment(
                        text=fields["text"],
                        speaker=fields["speaker"],
                        timestamp=float(fields["timestamp"]),
                        confidence=float(fields["confidence"]) if fields["confidence"] != "null" else None
                    )
                    segments.append(segment)
                except Exception as e:
                    self.logger.warning(
                        "Failed to parse transcript segment",
                        session_id=session_id,
                        message_id=message_id,
                        error=str(e)
                    )
                    continue
            
            self.logger.debug(
                "Transcript history retrieved",
                session_id=session_id,
                segment_count=len(segments)
            )
            
            return segments
            
        except Exception as e:
            self.logger.error(
                "Failed to get transcript history",
                **log_error(e, "get_transcript", "Redis"),
                session_id=session_id
            )
            return []
    
    # ===== PUB/SUB ДЛЯ ПРОМПТОВ =====
    
    async def publish_prompt_update(
        self, 
        session_id: str, 
        prompt_update: PromptUpdate
    ) -> bool:
        """
        Публикация обновления промпта
        
        Args:
            session_id: ID сессии
            prompt_update: Обновление промпта
            
        Returns:
            bool: Успешность операции
        """
        await self._ensure_connected()
        
        try:
            channel = self._get_key("prompt_update", session_id)
            
            # Публикуем обновление
            subscribers = await self._redis.publish(channel, prompt_update.json())
            
            self.logger.info(
                "Prompt update published",
                session_id=session_id,
                trigger=prompt_update.trigger_reason,
                subscribers=subscribers
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to publish prompt update",
                **log_error(e, "publish_prompt", "Redis"),
                session_id=session_id
            )
            return False
    
    async def subscribe_prompt_updates(
        self, 
        session_id: str
    ) -> AsyncGenerator[PromptUpdate, None]:
        """
        Подписка на обновления промптов
        
        Args:
            session_id: ID сессии
            
        Yields:
            PromptUpdate: Обновления промптов
        """
        await self._ensure_connected()
        
        try:
            channel = self._get_key("prompt_update", session_id)
            
            # Создаем pubsub если не существует
            if not self._pubsub:
                self._pubsub = self._redis.pubsub()
            
            await self._pubsub.subscribe(channel)
            
            self.logger.info(
                "Subscribed to prompt updates",
                session_id=session_id,
                channel=channel
            )
            
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        prompt_update = PromptUpdate.parse_raw(message["data"])
                        
                        self.logger.debug(
                            "Prompt update received",
                            session_id=session_id,
                            trigger=prompt_update.trigger_reason
                        )
                        
                        yield prompt_update
                        
                    except Exception as e:
                        self.logger.warning(
                            "Failed to parse prompt update",
                            session_id=session_id,
                            error=str(e)
                        )
                        continue
                        
        except Exception as e:
            self.logger.error(
                "Error in prompt updates subscription",
                **log_error(e, "subscribe_prompt", "Redis"),
                session_id=session_id
            )
        finally:
            if self._pubsub:
                try:
                    await self._pubsub.unsubscribe()
                except Exception:
                    pass
    
    # ===== МЕТРИКИ И МОНИТОРИНГ =====
    
    async def update_session_metrics(
        self, 
        session_id: str, 
        metrics: SessionMetrics
    ) -> bool:
        """
        Обновление метрик сессии
        
        Args:
            session_id: ID сессии
            metrics: Метрики сессии
            
        Returns:
            bool: Успешность операции
        """
        await self._ensure_connected()
        
        try:
            metrics_key = self._get_key("metrics", session_id)
            
            await self._redis.setex(
                metrics_key,
                settings.session_timeout,
                metrics.json()
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to update session metrics",
                **log_error(e, "update_metrics", "Redis"),
                session_id=session_id
            )
            return False
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Получение системных метрик
        
        Returns:
            Dict: Системные метрики
        """
        await self._ensure_connected()
        
        try:
            # Получаем информацию о Redis
            info = await self._redis.info()
            
            # Считаем активные сессии
            session_pattern = self._get_key("session", "*")
            session_keys = await self._redis.keys(session_pattern)
            
            # Статистика памяти
            memory_stats = {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0)
            }
            
            # Статистика соединений
            connection_stats = {
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
            
            metrics = {
                "active_sessions": len(session_keys),
                "redis_memory": memory_stats,
                "redis_connections": connection_stats,
                "timestamp": time.time()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                "Failed to get system metrics",
                **log_error(e, "get_metrics", "Redis")
            )
            return {}
    
    async def health_check(self) -> bool:
        """
        Проверка здоровья Redis сервиса
        
        Returns:
            bool: Статус здоровья
        """
        try:
            await self._ensure_connected()
            
            start_time = time.time()
            
            # Проверяем ping
            result = await self._redis.ping()
            
            if not result:
                return False
            
            # Проверяем операции записи/чтения
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            
            await self._redis.setex(test_key, 10, test_value)
            retrieved_value = await self._redis.get(test_key)
            await self._redis.delete(test_key)
            
            duration_ms = (time.time() - start_time) * 1000
            
            success = retrieved_value == test_value
            
            self.logger.info(
                "Redis health check completed",
                success=success,
                duration_ms=duration_ms
            )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Redis health check failed",
                **log_error(e, "health_check", "Redis")
            )
            return False
    
    # ===== КЕШИРОВАНИЕ =====
    
    async def cache_set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600
    ) -> bool:
        """
        Установка значения в кеш
        
        Args:
            key: Ключ кеша
            value: Значение для кеширования
            ttl: Время жизни в секундах
            
        Returns:
            bool: Успешность операции
        """
        await self._ensure_connected()
        
        try:
            cache_key = self._get_key("cache", key)
            
            # Сериализуем значение
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            result = await self._redis.setex(cache_key, ttl, serialized_value)
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(
                "Failed to set cache value",
                **log_error(e, "cache_set", "Redis"),
                key=key
            )
            return False
    
    async def cache_get(self, key: str) -> Optional[str]:
        """
        Получение значения из кеша
        
        Args:
            key: Ключ кеша
            
        Returns:
            Значение из кеша или None
        """
        await self._ensure_connected()
        
        try:
            cache_key = self._get_key("cache", key)
            value = await self._redis.get(cache_key)
            
            return value
            
        except Exception as e:
            self.logger.error(
                "Failed to get cache value",
                **log_error(e, "cache_get", "Redis"),
                key=key
            )
            return None