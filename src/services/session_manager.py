"""
Production-Ready Session Manager для Voice2Voice системы
КРИТИЧЕСКИ ВАЖНО: Управление жизненным циклом сессий с максимальной надежностью
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta

from ..core.config import settings
from ..core.models import (
    Session, SessionState, TranscriptSegment, PromptUpdate, 
    SessionMetrics, MessageRole
)
from ..core.exceptions import SessionError, RedisError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance
from .redis_service import RedisService


class SessionManager(LoggerMixin):
    """
    Production-Ready менеджер сессий с полным жизненным циклом
    Обеспечивает создание, управление и очистку сессий
    """
    
    def __init__(self, redis_service: RedisService):
        self.redis = redis_service
        self.max_concurrent_sessions = settings.max_concurrent_sessions
        self.session_timeout = settings.session_timeout
        self.max_history_length = settings.max_history_length
        
        # Активные сессии в памяти для быстрого доступа
        self._active_sessions: Dict[str, Session] = {}
        self._session_tasks: Dict[str, Set[asyncio.Task]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Блокировки для thread-safety
        self._sessions_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        
        # Статистика
        self._total_sessions_created = 0
        self._total_sessions_ended = 0
        
        # Запускаем периодическую очистку
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Запуск задачи периодической очистки"""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Периодическая очистка неактивных сессий"""
        while True:
            try:
                await asyncio.sleep(60)  # Очистка каждую минуту
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in periodic cleanup",
                    **log_error(e, "cleanup", "SessionManager")
                )
    
    async def _cleanup_expired_sessions(self):
        """Очистка истекших сессий"""
        async with self._cleanup_lock:
            try:
                current_time = time.time()
                expired_session_ids = []
                
                # Проверяем активные сессии в памяти
                for session_id, session in self._active_sessions.items():
                    # Проверяем время последней активности
                    if (current_time - session.metrics.last_activity) > self.session_timeout:
                        expired_session_ids.append(session_id)
                
                # Завершаем истекшие сессии
                for session_id in expired_session_ids:
                    await self._force_end_session(session_id, reason="expired")
                
                if expired_session_ids:
                    self.logger.info(
                        "Cleaned up expired sessions",
                        cleaned_count=len(expired_session_ids),
                        expired_sessions=expired_session_ids
                    )
                
            except Exception as e:
                self.logger.error(
                    "Error cleaning up expired sessions",
                    **log_error(e, "cleanup_expired", "SessionManager")
                )
    
    async def create_session(
        self, 
        session_id: Optional[str] = None,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Создание новой сессии
        
        Args:
            session_id: ID сессии (генерируется автоматически если не указан)
            client_info: Информация о клиенте
            
        Returns:
            Session: Созданная сессия
            
        Raises:
            SessionError: При ошибке создания сессии
        """
        async with self._sessions_lock:
            try:
                # Проверяем лимит сессий
                if len(self._active_sessions) >= self.max_concurrent_sessions:
                    raise SessionError(
                        f"Maximum concurrent sessions limit reached ({self.max_concurrent_sessions})"
                    )
                
                # Генерируем ID если не указан
                if not session_id:
                    session_id = str(uuid.uuid4())
                
                # Проверяем что сессия не существует
                if session_id in self._active_sessions:
                    raise SessionError(f"Session {session_id} already exists")
                
                start_time = time.time()
                
                # Создаем объект сессии
                session = Session(
                    id=session_id,
                    client_info=client_info or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Инициализируем метрики
                session.metrics = SessionMetrics(
                    session_id=session_id,
                    last_activity=time.time()
                )
                
                # Сохраняем в Redis
                await self.redis.create_session(session_id, {
                    "client_info": session.client_info
                })
                
                # Добавляем в активные сессии
                self._active_sessions[session_id] = session
                self._session_tasks[session_id] = set()
                
                self._total_sessions_created += 1
                
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    "Session created successfully",
                    **log_performance("create_session", duration_ms, "SessionManager"),
                    session_id=session_id,
                    total_active_sessions=len(self._active_sessions),
                    client_info=client_info
                )
                
                return session
                
            except Exception as e:
                if isinstance(e, SessionError):
                    raise
                    
                self.logger.error(
                    "Failed to create session",
                    **log_error(e, "create_session", "SessionManager"),
                    session_id=session_id
                )
                raise SessionError(f"Failed to create session: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Получение сессии по ID
        
        Args:
            session_id: ID сессии
            
        Returns:
            Session или None если не найдена
        """
        try:
            # Сначала проверяем в памяти
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                
                # Обновляем время последней активности
                session.metrics.last_activity = time.time()
                session.updated_at = datetime.now()
                
                return session
            
            # Если нет в памяти, проверяем в Redis
            session = await self.redis.get_session(session_id)
            if session:
                # Восстанавливаем в память если активна
                if session.state == SessionState.ACTIVE:
                    async with self._sessions_lock:
                        self._active_sessions[session_id] = session
                        self._session_tasks[session_id] = set()
                
                return session
            
            return None
            
        except Exception as e:
            self.logger.error(
                "Failed to get session",
                **log_error(e, "get_session", "SessionManager"),
                session_id=session_id
            )
            return None
    
    async def update_session(self, session: Session) -> bool:
        """
        Обновление сессии
        
        Args:
            session: Сессия для обновления
            
        Returns:
            bool: Успешность операции
        """
        try:
            session_id = session.id
            
            # Обновляем время
            session.updated_at = datetime.now()
            session.metrics.last_activity = time.time()
            
            # Обновляем в памяти
            if session_id in self._active_sessions:
                self._active_sessions[session_id] = session
            
            # Обновляем в Redis
            success = await self.redis.update_session(session)
            
            if success:
                # Обновляем метрики
                await self.redis.update_session_metrics(session_id, session.metrics)
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to update session",
                **log_error(e, "update_session", "SessionManager"),
                session_id=session.id
            )
            return False
    
    async def add_transcript_segment(
        self, 
        session_id: str, 
        segment: TranscriptSegment
    ) -> bool:
        """
        Добавление сегмента транскрипта в сессию
        
        Args:
            session_id: ID сессии
            segment: Сегмент транскрипта
            
        Returns:
            bool: Успешность операции
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                self.logger.warning(
                    "Attempt to add transcript to non-existent session",
                    session_id=session_id
                )
                return False
            
            # Добавляем в сессию
            session.add_transcript(segment)
            
            # Сохраняем в Redis Stream
            await self.redis.add_transcript_segment(session_id, segment)
            
            # Обновляем сессию
            await self.update_session(session)
            
            self.logger.debug(
                "Transcript segment added",
                session_id=session_id,
                speaker=segment.speaker.value,
                text_length=len(segment.text)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to add transcript segment",
                **log_error(e, "add_transcript", "SessionManager"),
                session_id=session_id
            )
            return False
    
    async def update_prompt(
        self, 
        session_id: str, 
        prompt_update: PromptUpdate
    ) -> bool:
        """
        Обновление промпта сессии
        
        Args:
            session_id: ID сессии
            prompt_update: Обновление промпта
            
        Returns:
            bool: Успешность операции
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                self.logger.warning(
                    "Attempt to update prompt for non-existent session",
                    session_id=session_id
                )
                return False
            
            # Проверяем, не совпадает ли новый промпт с текущим
            if session.current_prompt == prompt_update.new_prompt:
                self.logger.debug(
                    "Prompt update skipped - same as current",
                    session_id=session_id,
                    prompt=prompt_update.new_prompt[:50] + "..."
                )
                return True  # Возвращаем True, т.к. промпт уже актуален
            
            # Обновляем промпт
            session.update_prompt(prompt_update)
            
            # Публикуем обновление
            await self.redis.publish_prompt_update(session_id, prompt_update)
            
            # Обновляем сессию
            await self.update_session(session)
            
            self.logger.info(
                "Session prompt updated",
                session_id=session_id,
                old_prompt=prompt_update.old_prompt[:50] + "...",
                new_prompt=prompt_update.new_prompt[:50] + "...",
                trigger=prompt_update.trigger_reason
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to update session prompt",
                **log_error(e, "update_prompt", "SessionManager"),
                session_id=session_id
            )
            return False
    
    async def end_session(
        self, 
        session_id: str,
        reason: str = "user_disconnect"
    ) -> bool:
        """
        Завершение сессии
        
        Args:
            session_id: ID сессии
            reason: Причина завершения
            
        Returns:
            bool: Успешность операции
        """
        async with self._sessions_lock:
            try:
                session = await self.get_session(session_id)
                if not session:
                    self.logger.warning(
                        "Attempt to end non-existent session",
                        session_id=session_id
                    )
                    return False
                
                start_time = time.time()
                
                # Обновляем состояние
                session.state = SessionState.ENDED
                session.updated_at = datetime.now()
                
                # Отменяем связанные задачи
                if session_id in self._session_tasks:
                    tasks = self._session_tasks[session_id]
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    del self._session_tasks[session_id]
                
                # Сохраняем финальное состояние в Redis
                await self.redis.update_session(session)
                
                # Удаляем из активных сессий
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
                
                self._total_sessions_ended += 1
                
                # Вычисляем продолжительность сессии
                session_duration = (session.updated_at - session.created_at).total_seconds()
                
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    "Session ended successfully",
                    **log_performance("end_session", duration_ms, "SessionManager"),
                    session_id=session_id,
                    reason=reason,
                    session_duration_seconds=session_duration,
                    total_messages=session.metrics.total_user_messages + session.metrics.total_ai_responses,
                    prompt_changes=session.metrics.prompt_changes,
                    remaining_active_sessions=len(self._active_sessions)
                )
                
                return True
                
            except Exception as e:
                self.logger.error(
                    "Failed to end session",
                    **log_error(e, "end_session", "SessionManager"),
                    session_id=session_id,
                    reason=reason
                )
                return False
    
    async def _force_end_session(self, session_id: str, reason: str = "forced") -> None:
        """Принудительное завершение сессии (для внутреннего использования)"""
        try:
            await self.end_session(session_id, reason)
        except Exception as e:
            self.logger.error(
                "Failed to force end session",
                **log_error(e, "force_end_session", "SessionManager"),
                session_id=session_id
            )
    
    async def pause_session(self, session_id: str) -> bool:
        """
        Приостановка сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            bool: Успешность операции
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.state = SessionState.PAUSED
            success = await self.update_session(session)
            
            if success:
                self.logger.info(
                    "Session paused",
                    session_id=session_id
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to pause session",
                **log_error(e, "pause_session", "SessionManager"),
                session_id=session_id
            )
            return False
    
    async def resume_session(self, session_id: str) -> bool:
        """
        Возобновление сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            bool: Успешность операции
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.state = SessionState.ACTIVE
            session.metrics.last_activity = time.time()
            success = await self.update_session(session)
            
            if success:
                self.logger.info(
                    "Session resumed",
                    session_id=session_id
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Failed to resume session",
                **log_error(e, "resume_session", "SessionManager"),
                session_id=session_id
            )
            return False
    
    async def register_session_task(
        self, 
        session_id: str, 
        task: asyncio.Task
    ) -> None:
        """
        Регистрация задачи для сессии (для отслеживания и очистки)
        
        Args:
            session_id: ID сессии
            task: Задача для регистрации
        """
        if session_id in self._session_tasks:
            self._session_tasks[session_id].add(task)
            
            # Добавляем callback для автоматической очистки завершенных задач
            def cleanup_task(t):
                if session_id in self._session_tasks:
                    self._session_tasks[session_id].discard(t)
            
            task.add_done_callback(cleanup_task)
    
    def get_active_sessions_count(self) -> int:
        """
        Получение количества активных сессий
        
        Returns:
            int: Количество активных сессий
        """
        return len(self._active_sessions)
    
    def get_active_session_ids(self) -> List[str]:
        """
        Получение списка ID активных сессий
        
        Returns:
            List[str]: Список ID активных сессий
        """
        return list(self._active_sessions.keys())
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Получение статистики сессий
        
        Returns:
            Dict: Статистика сессий
        """
        active_sessions = len(self._active_sessions)
        
        # Подсчитываем состояния
        state_counts = {}
        for session in self._active_sessions.values():
            state = session.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Средняя продолжительность активных сессий
        total_duration = 0
        now = datetime.now()
        for session in self._active_sessions.values():
            duration = (now - session.created_at).total_seconds()
            total_duration += duration
        
        avg_duration = total_duration / active_sessions if active_sessions > 0 else 0
        
        return {
            "active_sessions": active_sessions,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "total_sessions_created": self._total_sessions_created,
            "total_sessions_ended": self._total_sessions_ended,
            "session_states": state_counts,
            "average_session_duration_seconds": avg_duration,
            "utilization_percent": (active_sessions / self.max_concurrent_sessions) * 100
        }
    
    async def health_check(self) -> bool:
        """
        Проверка здоровья Session Manager
        
        Returns:
            bool: Статус здоровья
        """
        try:
            # Проверяем Redis соединение
            redis_healthy = await self.redis.health_check()
            if not redis_healthy:
                return False
            
            # Проверяем что количество сессий в пределах нормы
            active_count = self.get_active_sessions_count()
            if active_count > self.max_concurrent_sessions:
                self.logger.warning(
                    "Too many active sessions",
                    active_sessions=active_count,
                    max_sessions=self.max_concurrent_sessions
                )
                return False
            
            # Проверяем что cleanup task работает
            if self._cleanup_task and self._cleanup_task.done():
                self.logger.warning("Cleanup task is not running")
                self._start_cleanup_task()
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Session manager health check failed",
                **log_error(e, "health_check", "SessionManager")
            )
            return False
    
    async def shutdown(self) -> None:
        """
        Корректное завершение работы Session Manager
        """
        try:
            self.logger.info("Shutting down Session Manager...")
            
            # Останавливаем cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Завершаем все активные сессии
            session_ids = list(self._active_sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id, reason="shutdown")
            
            self.logger.info(
                "Session Manager shutdown completed",
                sessions_ended=len(session_ids)
            )
            
        except Exception as e:
            self.logger.error(
                "Error during Session Manager shutdown",
                **log_error(e, "shutdown", "SessionManager")
            )