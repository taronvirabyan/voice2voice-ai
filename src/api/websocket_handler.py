"""
Production-Ready WebSocket Handler для Voice2Voice системы
КРИТИЧЕСКИ ВАЖНО: Центральный компонент для real-time обработки аудио диалогов
"""

import asyncio
import json
import time
import uuid
import base64
from typing import Dict, Optional, Set, Any, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect
import traceback

from ..core.config import settings
from ..core.models import (
    Session, SessionState, TranscriptSegment, MessageRole, 
    VoiceResponse, AudioChunk, PromptUpdate
)
from ..core.exceptions import (
    WebSocketError, SessionError, AudioProcessingError,
    SaluteSpeechError, GeminiError
)
from ..core.logging import LoggerMixin, log_request, log_error, log_performance
from ..core.smart_start_manager import SmartStartManager
from ..core.voice_activity_detector import VoiceActivityDetector

# Импорты сервисов
from ..services.stt_service import SaluteSpeechSTT
from ..services.tts_service import SaluteSpeechTTS
from ..services.gemini_voice_ai_service import GeminiVoiceAIService
from ..services.gemini_moderator_service import GeminiModeratorService
from ..services.robust_gemini_service import RobustGeminiService
from ..services.redis_service import RedisService
from ..services.session_manager import SessionManager


class ConnectionManager(LoggerMixin):
    """
    Менеджер WebSocket соединений
    Управляет активными соединениями и рассылкой сообщений
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, str] = {}  # websocket_id -> session_id
        self._connection_lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str) -> str:
        """
        Подключение нового WebSocket клиента
        
        Args:
            websocket: WebSocket соединение
            session_id: ID сессии
            
        Returns:
            str: ID соединения
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        
        async with self._connection_lock:
            self.active_connections[connection_id] = websocket
            self.connection_sessions[connection_id] = session_id
        
        self.logger.info(
            "WebSocket connected",
            connection_id=connection_id,
            session_id=session_id,
            total_connections=len(self.active_connections)
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str) -> Optional[str]:
        """
        Отключение WebSocket клиента
        
        Args:
            connection_id: ID соединения
            
        Returns:
            str: ID сессии или None
        """
        session_id = None
        
        async with self._connection_lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            if connection_id in self.connection_sessions:
                session_id = self.connection_sessions[connection_id]
                del self.connection_sessions[connection_id]
        
        if session_id:
            self.logger.info(
                "WebSocket disconnected",
                connection_id=connection_id,
                session_id=session_id,
                remaining_connections=len(self.active_connections)
            )
        
        return session_id
    
    async def send_audio(self, session_id: str, audio_data: bytes) -> bool:
        """
        Отправка аудио данных клиенту
        
        Args:
            session_id: ID сессии
            audio_data: Аудио данные
            
        Returns:
            bool: Успешность отправки
        """
        try:
            # Ищем соединение по session_id
            connection_id = None
            for conn_id, sess_id in self.connection_sessions.items():
                if sess_id == session_id:
                    connection_id = conn_id
                    break
            
            if not connection_id or connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            
            # Проверяем состояние WebSocket перед отправкой
            if hasattr(websocket, 'client_state') and websocket.client_state.name != 'CONNECTED':
                self.logger.warning(
                    "WebSocket not in CONNECTED state",
                    session_id=session_id,
                    state=websocket.client_state.name if hasattr(websocket, 'client_state') else 'unknown'
                )
                return False
                
            await websocket.send_bytes(audio_data)
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to send audio",
                **log_error(e, "send_audio", "ConnectionManager"),
                session_id=session_id,
                audio_size=len(audio_data)
            )
            return False
    
    async def send_json(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Отправка JSON сообщения клиенту
        
        Args:
            session_id: ID сессии
            data: JSON данные
            
        Returns:
            bool: Успешность отправки
        """
        try:
            # Ищем соединение по session_id
            connection_id = None
            for conn_id, sess_id in self.connection_sessions.items():
                if sess_id == session_id:
                    connection_id = conn_id
                    break
            
            if not connection_id or connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            
            # Проверяем состояние WebSocket перед отправкой
            if hasattr(websocket, 'client_state') and websocket.client_state.name != 'CONNECTED':
                self.logger.warning(
                    "WebSocket not in CONNECTED state for JSON",
                    session_id=session_id,
                    state=websocket.client_state.name if hasattr(websocket, 'client_state') else 'unknown',
                    message_type=data.get('type', 'unknown')
                )
                return False
                
            await websocket.send_json(data)
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to send JSON",
                **log_error(e, "send_json", "ConnectionManager"),
                session_id=session_id,
                data_keys=list(data.keys()) if data else []
            )
            return False
    
    def get_connection_count(self) -> int:
        """Получение количества активных соединений"""
        return len(self.active_connections)


class AudioPipeline(LoggerMixin):
    """
    Production-Ready аудио пайплайн для обработки Voice2Voice диалогов
    Объединяет STT -> AI -> TTS в единый поток
    """
    
    def __init__(self, services: Dict[str, Any]):
        self.stt = services.get("stt")  # Optional in development
        self.tts = services.get("tts")  # Optional in development
        self.voice_ai = services.get("voice_ai")  # Optional if quota exceeded
        self.moderator = services.get("moderator")  # Optional if quota exceeded
        self.redis = services["redis"]
        self.session_manager = services["session_manager"]
        self.connection_manager = services["connection_manager"]
        self.smart_start_manager = services.get("smart_start_manager")  # Optional
        self.vad = services.get("vad")  # Optional
        
        # Настройки производительности
        self.max_concurrent_processing = 3
        self.audio_buffer_size = settings.chunk_size * 4
        
    async def process_audio_stream(
        self, 
        session_id: str,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> None:
        """
        Основной пайплайн обработки аудио потока
        Работает в режиме text-only если STT/TTS недоступны
        
        Args:
            session_id: ID сессии
            audio_stream: Поток аудио данных от клиента
        """
        self.logger.info(
            "Starting audio pipeline",
            session_id=session_id,
            stt_available=self.stt is not None,
            tts_available=self.tts is not None
        )
        
        try:
            # Получаем сессию
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise SessionError(f"Session {session_id} not found")
            
            # Если STT/TTS недоступны, работаем в text simulation режиме
            if self.stt is None or self.tts is None:
                missing_services = []
                if self.stt is None:
                    missing_services.append("STT")
                if self.tts is None:
                    missing_services.append("TTS")
                    
                await self.connection_manager.send_json(session_id, {
                    "type": "info", 
                    "message": f"⚠️ {', '.join(missing_services)} service(s) unavailable. Running AI simulation mode.",
                    "mode": "text_simulation",
                    "missing_services": missing_services
                })
                
                # Запускаем text-based симуляцию voice2voice диалога
                await self._run_text_simulation(session_id, audio_stream)
                return
            
            # Уведомляем о production voice2voice режиме
            await self.connection_manager.send_json(session_id, {
                "type": "voice_mode_started",
                "message": "🎙️ PRODUCTION Voice2Voice режим активен (Whisper + ElevenLabs)",
                "services": {
                    "stt": type(self.stt).__name__,
                    "tts": type(self.tts).__name__
                }
            })
            
            # Полнофункциональный voice2voice режим
            try:
                # Запускаем обработку STT потока
                self.logger.info(
                    "Starting STT transcription",
                    session_id=session_id
                )
                
                transcript_count = 0
                
                # Основной цикл обработки транскрипций БЕЗ перезапуска
                # Whisper STT сервис сам обрабатывает непрерывный поток
                async for transcript_segment in self.stt.transcribe_stream(
                    audio_stream, 
                    session_id
                ):
                    transcript_count += 1
                    self.logger.info(
                        f"Processing transcript #{transcript_count}",
                        session_id=session_id,
                        text_preview=transcript_segment.text[:50] if transcript_segment.text else ""
                    )
                    # Smart Start: обрабатываем текст для оптимизации
                    if hasattr(self, 'smart_start_manager'):
                        await self.smart_start_manager.process_transcript_text(
                            session_id, 
                            transcript_segment.text,
                            confidence=transcript_segment.confidence if hasattr(transcript_segment, 'confidence') else 1.0
                        )
                    
                    # Отправляем транскрипцию клиенту
                    await self.connection_manager.send_json(session_id, {
                        "type": "transcription",
                        "text": transcript_segment.text,
                        "speaker": transcript_segment.speaker.value,
                        "confidence": transcript_segment.confidence,
                        "timestamp": transcript_segment.timestamp
                    })
                    
                    # Добавляем транскрипт в сессию
                    await self.session_manager.add_transcript_segment(
                        session_id, 
                        transcript_segment
                    )
                    
                    # Получаем обновленную сессию
                    session = await self.session_manager.get_session(session_id)
                    if not session or not session.is_active():
                        self.logger.warning(
                            "Session became inactive, stopping audio pipeline",
                            session_id=session_id,
                            session_exists=session is not None,
                            session_active=session.is_active() if session else False
                        )
                        return  # Выходим из всей функции
                    
                    # КРИТИЧЕСКИ ВАЖНО: Сначала даем модератору проанализировать диалог
                    # и обновить промпт, если нужно
                    if self.moderator:
                        self.logger.debug(
                            "🔍 Analyzing conversation for prompt update",
                            session_id=session_id,
                            current_prompt_preview=session.current_prompt[:50] + "..." if len(session.current_prompt) > 50 else session.current_prompt,
                            transcript_count=len(session.transcript_history),
                            last_user_text=transcript_segment.text
                        )
                        
                        try:
                            # Анализируем текущий диалог
                            prompt_update = await self.moderator.analyze_with_fallback(
                                transcript_history=session.transcript_history,
                                current_prompt=session.current_prompt,
                                session_id=session.id
                            )
                        except Exception as e:
                            self.logger.error(
                                "❌ Moderator analysis failed",
                                error=str(e),
                                error_type=type(e).__name__,
                                session_id=session_id
                            )
                            prompt_update = None
                        
                        if prompt_update:
                            self.logger.info(
                                "🎯 PROMPT UPDATE DETECTED!",
                                old_prompt_preview=session.current_prompt[:50] + "..." if len(session.current_prompt) > 50 else session.current_prompt,
                                new_prompt_preview=prompt_update.new_prompt[:50] + "..." if len(prompt_update.new_prompt) > 50 else prompt_update.new_prompt,
                                trigger_keywords=prompt_update.trigger_keywords,
                                confidence=prompt_update.confidence
                            )
                            
                            # Обновляем промпт в сессии
                            success = await self.session_manager.update_prompt(
                                session_id,
                                prompt_update
                            )
                            
                            if success:
                                # Уведомляем клиента о смене промпта
                                send_success = await self.connection_manager.send_json(session_id, {
                                    "type": "prompt_update",
                                    "data": {
                                        "new_prompt": prompt_update.new_prompt,
                                        "trigger": prompt_update.trigger_reason,
                                        "confidence": prompt_update.confidence
                                    }
                                })
                                
                                if not send_success:
                                    self.logger.error(
                                        "❌ CRITICAL: Failed to send prompt_update to client!",
                                        session_id=session_id,
                                        new_prompt=prompt_update.new_prompt[:50] + "...",
                                        trigger=prompt_update.trigger_reason
                                    )
                                else:
                                    self.logger.info(
                                        "✅ Prompt update notification sent to client successfully",
                                        session_id=session_id,
                                        message_type="prompt_update"
                                    )
                                
                                # Получаем обновленную сессию с новым промптом
                                session = await self.session_manager.get_session(session_id)
                                
                                self.logger.info(
                                    "✅ Prompt updated before AI response",
                                    session_id=session_id,
                                    new_prompt=prompt_update.new_prompt[:50] + "...",
                                    trigger=prompt_update.trigger_reason
                                )
                        else:
                            self.logger.debug(
                                "📝 No prompt update needed",
                                session_id=session_id,
                                current_prompt=session.current_prompt[:50] + "..."
                            )
                    else:
                        self.logger.warning(
                            "⚠️ Moderator not initialized, skipping prompt analysis",
                            session_id=session_id
                        )
                    
                    # Smart Start: отмечаем начало AI обработки
                    if hasattr(self, 'smart_start_manager'):
                        await self.smart_start_manager.mark_ai_processing_started(session_id)
                    
                    # Генерируем ответ AI с актуальным промптом
                    ai_response = await self._generate_ai_response(
                        session, 
                        transcript_segment.text
                    )
                    
                    if ai_response:
                        # Отправляем ответ AI клиенту
                        await self.connection_manager.send_json(session_id, {
                            "type": "ai_response",
                            "text": ai_response,
                            "timestamp": time.time()
                        })
                        
                        # Создаем сегмент ответа AI
                        ai_segment = TranscriptSegment(
                            text=ai_response,
                            speaker=MessageRole.ASSISTANT,
                            timestamp=time.time()
                        )
                        
                        # Добавляем в сессию
                        await self.session_manager.add_transcript_segment(
                            session_id,
                            ai_segment
                        )
                        
                        # Синтезируем и отправляем аудио
                        await self._synthesize_and_send_audio(
                            session_id,
                            ai_response,
                            self.tts
                        )
                        
                        # Smart Start: сбрасываем состояние для следующей фразы
                        if hasattr(self, 'smart_start_manager'):
                            await self.smart_start_manager.reset_session_state(session_id)
                        
                        # VAD: сбрасываем состояние для следующей фразы
                        if hasattr(self, 'vad'):
                            await self.vad.reset_session(session_id)
                
                # STT поток завершился нормально
                self.logger.info(
                    f"✅ STT stream completed normally",
                    session_id=session_id,
                    total_transcripts=transcript_count
                )
                
                # Проверяем, активна ли еще сессия
                session = await self.session_manager.get_session(session_id)
                if session and session.is_active():
                    self.logger.warning(
                        "⚠️ STT stream ended but session still active",
                        session_id=session_id
                    )
                
            except asyncio.CancelledError:
                self.logger.info(
                    "STT stream cancelled",
                    session_id=session_id
                )
            except Exception as stt_error:
                self.logger.error(
                    "STT processing error",
                    **log_error(stt_error, "stt_stream", "AudioPipeline"),
                    session_id=session_id
                )
                await self.connection_manager.send_json(session_id, {
                    "type": "error",
                    "message": f"Speech recognition error: {str(stt_error)}",
                    "code": "STT_ERROR"
                })
                
        except asyncio.CancelledError:
            self.logger.info(
                "Audio pipeline cancelled",
                session_id=session_id
            )
        except Exception as e:
            self.logger.error(
                "Audio pipeline error",
                **log_error(e, "audio_pipeline", "AudioPipeline"),
                session_id=session_id
            )
            
            # Отправляем ошибку клиенту
            await self.connection_manager.send_json(session_id, {
                "type": "error",
                "message": "Audio processing error",
                "code": "PIPELINE_ERROR"
            })
    
    async def _run_text_simulation(
        self, 
        session_id: str,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> None:
        """
        Симуляция voice2voice диалога в text режиме
        Когда STT/TTS недоступны, но AI работает
        
        Args:
            session_id: ID сессии
            audio_stream: Поток аудио (игнорируется в симуляции)
        """
        try:
            self.logger.info(
                "Starting text simulation mode",
                session_id=session_id,
                voice_ai_available=self.voice_ai is not None
            )
            
            # Если AI тоже недоступен, не можем симулировать
            if self.voice_ai is None:
                await self.connection_manager.send_json(session_id, {
                    "type": "error",
                    "message": "AI services unavailable. Cannot run simulation.",
                    "mode": "no_simulation"
                })
                return
            
            # Отправляем инструкцию пользователю
            await self.connection_manager.send_json(session_id, {
                "type": "simulation_start",
                "message": "🎭 TEXT SIMULATION MODE: Отправьте JSON с текстом для диалога",
                "instruction": {
                    "format": "JSON",
                    "example": '{"type": "user_message", "text": "Привет! Как дела?"}',
                    "note": "Система симулирует voice2voice диалог в текстовом режиме"
                }
            })
            
            # Запускаем task для чтения аудио потока (игнорируем данные)
            async def consume_audio_stream():
                try:
                    async for _ in audio_stream:
                        pass  # Просто читаем и игнорируем аудио данные
                except Exception:
                    pass  # Игнорируем ошибки аудио потока в симуляции
            
            # Запускаем в фоне
            audio_task = asyncio.create_task(consume_audio_stream())
            
            # Симуляция автоматического диалога каждые 8 секунд
            simulation_messages = [
                "Привет! Как дела?",
                "У меня есть собака породы лабрадор", 
                "Недавно думаю о покупке ошейника",
                "Может быть есть какие-то рекомендации?",
                "А что вы думаете о бизнес возможностях?"
            ]
            
            for i, message in enumerate(simulation_messages):
                await asyncio.sleep(8)  # Пауза между сообщениями
                
                # Проверяем активность сессии
                session = await self.session_manager.get_session(session_id)
                if not session or not session.is_active():
                    break
                
                # Симулируем пользовательское сообщение
                await self.connection_manager.send_json(session_id, {
                    "type": "simulated_user_input",
                    "text": message,
                    "step": i + 1,
                    "total_steps": len(simulation_messages)
                })
                
                # Создаем transcript segment для пользователя
                user_segment = TranscriptSegment(
                    text=message,
                    speaker=MessageRole.USER,
                    timestamp=time.time()
                )
                
                # Добавляем в сессию
                await self.session_manager.add_transcript_segment(
                    session_id,
                    user_segment
                )
                
                # Получаем актуальную сессию
                session = await self.session_manager.get_session(session_id)
                
                # КРИТИЧЕСКИ ВАЖНО: Сначала анализируем модератором
                if self.moderator:
                    prompt_update = await self.moderator.analyze_conversation(
                        transcript_history=session.transcript_history,
                        current_prompt=session.current_prompt,
                        session_id=session.id
                    )
                    
                    if prompt_update:
                        # Обновляем промпт
                        success = await self.session_manager.update_prompt(
                            session_id,
                            prompt_update
                        )
                        
                        if success:
                            # Получаем обновленную сессию
                            session = await self.session_manager.get_session(session_id)
                            
                            # Уведомляем о смене промпта (используем стандартный формат)
                            send_success = await self.connection_manager.send_json(session_id, {
                                "type": "prompt_update",
                                "data": {
                                    "new_prompt": prompt_update.new_prompt,
                                    "trigger": prompt_update.trigger_reason,
                                    "confidence": prompt_update.confidence
                                }
                            })
                            
                            if not send_success:
                                self.logger.error(
                                    "❌ CRITICAL: Failed to send prompt_update in simulation!",
                                    session_id=session_id
                                )
                            else:
                                self.logger.info(
                                    "✅ Prompt update sent in simulation mode",
                                    session_id=session_id
                                )
                
                # Генерируем ответ AI с актуальным промптом
                ai_response = await self._generate_ai_response(session, message)
                
                if ai_response:
                    # Создаем сегмент ответа AI
                    ai_segment = TranscriptSegment(
                        text=ai_response,
                        speaker=MessageRole.ASSISTANT,
                        timestamp=time.time()
                    )
                    
                    # Добавляем в сессию
                    await self.session_manager.add_transcript_segment(
                        session_id,
                        ai_segment
                    )
                    
                    # Отправляем ответ AI
                    await self.connection_manager.send_json(session_id, {
                        "type": "simulated_ai_response",
                        "text": ai_response,
                        "step": i + 1,
                        "note": "В реальном режиме это был бы синтезированный голос"
                    })
                
                await asyncio.sleep(2)  # Пауза после ответа AI
            
            # Завершение симуляции
            await self.connection_manager.send_json(session_id, {
                "type": "simulation_complete",
                "message": "🎭 Симуляция завершена! В production это был бы полный voice2voice диалог.",
                "summary": "Система продемонстрировала корректную работу AI и moderator компонентов"
            })
            
            # Завершаем audio task
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass
            
        except asyncio.CancelledError:
            self.logger.info(
                "Text simulation cancelled",
                session_id=session_id
            )
        except Exception as e:
            self.logger.error(
                "Text simulation error",
                **log_error(e, "text_simulation", "AudioPipeline"),
                session_id=session_id
            )
            
            await self.connection_manager.send_json(session_id, {
                "type": "error",
                "message": "Simulation error occurred",
                "code": "SIMULATION_ERROR"
            })
    
    async def _generate_ai_response(
        self, 
        session: Session, 
        user_message: str
    ) -> Optional[str]:
        """
        Генерация ответа AI на основе сообщения пользователя
        
        Args:
            session: Сессия диалога
            user_message: Сообщение пользователя
            
        Returns:
            str: Ответ AI или None при ошибке
        """
        try:
            start_time = time.time()
            
            # Проверяем наличие AI сервиса
            if not self.voice_ai:
                self.logger.error(
                    "🚨 CRITICAL: AI service not available - $1000 demo at risk!",
                    session_id=session.id,
                    available_services=list(self.__dict__.keys())
                )
                # Пытаемся использовать простой ответ для демонстрации
                demo_responses = [
                    "Привет! Я виртуальный помощник. Чем могу помочь?",
                    "Интересный вопрос! Расскажите подробнее.",
                    "Понимаю вас. Что еще вы хотели бы обсудить?",
                    "Спасибо за ваше сообщение. Продолжайте, я слушаю."
                ]
                import random
                return random.choice(demo_responses)
            
            # Получаем историю диалога
            recent_history = session.get_recent_history(10)
            
            # Генерируем ответ
            response = await self.voice_ai.generate_response(
                user_message=user_message,
                current_prompt=session.current_prompt,
                conversation_history=recent_history,
                session_id=session.id
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                "AI response generated",
                **log_performance("ai_response", duration_ms, "AudioPipeline"),
                session_id=session.id,
                response_length=len(response) if response else 0
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Failed to generate AI response",
                **log_error(e, "ai_response", "AudioPipeline"),
                session_id=session.id,
                user_message=user_message[:50] + "..."
            )
            return "Извините, произошла ошибка. Можете повторить?"
    
    async def _synthesize_and_send_audio(
        self,
        session_id: str,
        text: str,
        tts_service: Any
    ) -> None:
        """
        Синтез речи и отправка аудио клиенту
        
        Args:
            session_id: ID сессии
            text: Текст для синтеза
            tts_service: Сервис TTS
        """
        try:
            start_time = time.time()
            
            # Определяем формат аудио в зависимости от TTS сервиса
            # MockTTS отправляет WAV, ElevenLabs - MP3
            audio_format = "wav" if "MockTTS" in str(type(tts_service)) else "mp3"
            self.logger.info(
                f"TTS format detected: {audio_format}",
                session_id=session_id,
                tts_service_type=str(type(tts_service))
            )
            
            # Буфер для объединения мелких чанков
            # Параметры оптимизированы для баланса между плавностью и скоростью отклика
            chunk_buffer = []
            buffer_size = 0
            target_buffer_size = 32 * 1024  # 32 КБ - оптимальный баланс (2-3 чанка)
            chunk_index = 0
            is_first_chunk_batch = True  # Флаг для первой партии чанков
            
            # Streaming синтез и отправка
            async for audio_chunk in tts_service.synthesize_stream(
                text=text,
                session_id=session_id
            ):
                # Накапливаем чанки
                chunk_buffer.append(audio_chunk)
                buffer_size += len(audio_chunk)
                
                # Для первой партии используем меньший буфер для быстрого старта
                current_target = 16384 if is_first_chunk_batch else target_buffer_size  # 16 КБ для первого, 32 КБ для остальных
                
                # Отправляем когда накопили достаточно
                if buffer_size >= current_target:
                    combined_chunk = b''.join(chunk_buffer)
                    audio_base64 = base64.b64encode(combined_chunk).decode('utf-8')
                    
                    success = await self.connection_manager.send_json(session_id, {
                        "type": "audio_chunk",
                        "audio": audio_base64,
                        "size": len(combined_chunk),
                        "format": audio_format,  # Используем определенный формат
                        "chunk_index": chunk_index,
                        "buffer_optimized": True,
                        "is_first_batch": is_first_chunk_batch
                    })
                    
                    if not success:
                        self.logger.warning(
                            "Failed to send audio chunk",
                            session_id=session_id
                        )
                        break
                    
                    # Сброс буфера
                    chunk_buffer = []
                    buffer_size = 0
                    chunk_index += 1
                    is_first_chunk_batch = False  # Сбрасываем флаг после первой отправки
            
            # Отправляем остатки если есть
            if chunk_buffer:
                combined_chunk = b''.join(chunk_buffer)
                audio_base64 = base64.b64encode(combined_chunk).decode('utf-8')
                
                self.logger.info(
                    "📍 Sending final chunk with remaining buffer",
                    session_id=session_id,
                    chunk_index=chunk_index,
                    buffer_size=len(combined_chunk)
                )
                
                await self.connection_manager.send_json(session_id, {
                    "type": "audio_chunk",
                    "audio": audio_base64,
                    "size": len(combined_chunk),
                    "format": audio_format,
                    "chunk_index": chunk_index,
                    "is_last": True,
                    "buffer_optimized": True
                })
            else:
                # КРИТИЧНО: Если буфер пуст, все равно отправляем сигнал завершения
                self.logger.info(
                    "📍 Sending empty chunk with is_last flag",
                    session_id=session_id,
                    chunk_index=chunk_index
                )
                await self.connection_manager.send_json(session_id, {
                    "type": "audio_chunk",
                    "audio": "",  # Пустой чанк
                    "size": 0,
                    "format": audio_format,
                    "chunk_index": chunk_index,
                    "is_last": True,
                    "buffer_optimized": True
                })
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                "Audio synthesis and streaming completed",
                **log_performance("audio_synthesis", duration_ms, "AudioPipeline"),
                session_id=session_id,
                text_length=len(text)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to synthesize and send audio",
                **log_error(e, "audio_synthesis", "AudioPipeline"),
                session_id=session_id,
                text=text[:50] + "..."
            )


class ModeratorPipeline(LoggerMixin):
    """
    Пайплайн модератора для анализа диалогов и обновления промптов
    """
    
    def __init__(self, services: Dict[str, Any]):
        self.moderator = services.get("moderator")  # Optional if quota exceeded
        self.session_manager = services["session_manager"]
        self.connection_manager = services["connection_manager"]
        self.redis = services["redis"]
        
        # Настройки модератора
        self.analysis_interval = 5  # Анализ каждые 5 секунд
        self.min_messages_for_analysis = 1  # КРИТИЧЕСКИ ВАЖНО: Анализируем с первого сообщения
    
    async def run_moderator_for_session(self, session_id: str) -> None:
        """
        Запуск модератора для конкретной сессии
        
        Args:
            session_id: ID сессии
        """
        self.logger.info(
            "Starting moderator for session",
            session_id=session_id
        )
        
        try:
            while True:
                await asyncio.sleep(self.analysis_interval)
                
                # Получаем сессию
                session = await self.session_manager.get_session(session_id)
                if not session or not session.is_active():
                    self.logger.info(
                        "Session inactive, stopping moderator",
                        session_id=session_id
                    )
                    break
                
                # Проверяем достаточно ли сообщений для анализа
                if len(session.transcript_history) < self.min_messages_for_analysis:
                    self.logger.debug(
                        "Not enough messages for analysis",
                        session_id=session_id,
                        current_count=len(session.transcript_history),
                        required_count=self.min_messages_for_analysis
                    )
                    continue
                
                # Анализируем диалог
                self.logger.info(
                    "Starting conversation analysis",
                    session_id=session_id,
                    message_count=len(session.transcript_history),
                    current_prompt_preview=session.current_prompt[:50] + "..." if len(session.current_prompt) > 50 else session.current_prompt
                )
                prompt_update = await self._analyze_conversation(session)
                
                if prompt_update:
                    # Обновляем промпт
                    success = await self.session_manager.update_prompt(
                        session_id,
                        prompt_update
                    )
                    
                    if success:
                        # Уведомляем клиента
                        send_success = await self.connection_manager.send_json(session_id, {
                            "type": "prompt_update",
                            "data": {
                                "new_prompt": prompt_update.new_prompt,
                                "trigger": prompt_update.trigger_reason,
                                "confidence": prompt_update.confidence
                            }
                        })
                        
                        if not send_success:
                            self.logger.error(
                                "❌ CRITICAL: Failed to send prompt_update from ModeratorPipeline!",
                                session_id=session_id,
                                new_prompt=prompt_update.new_prompt[:50] + "..."
                            )
                        else:
                            self.logger.info(
                                "✅ Prompt update sent from ModeratorPipeline",
                                session_id=session_id
                            )
                
        except asyncio.CancelledError:
            self.logger.info(
                "Moderator cancelled for session",
                session_id=session_id
            )
        except Exception as e:
            self.logger.error(
                "Moderator error",
                **log_error(e, "moderator", "ModeratorPipeline"),
                session_id=session_id
            )
    
    async def _analyze_conversation(self, session: Session) -> Optional[PromptUpdate]:
        """
        Анализ разговора для определения необходимости смены промпта
        
        Args:
            session: Сессия диалога
            
        Returns:
            PromptUpdate или None
        """
        try:
            start_time = time.time()
            
            # Проверяем наличие модератора
            if not self.moderator:
                self.logger.debug(
                    "Moderator not available (likely quota exceeded)",
                    session_id=session.id
                )
                return None
            
            # Анализируем диалог с fallback на ключевые слова
            result = await self.moderator.analyze_with_fallback(
                transcript_history=session.transcript_history,
                current_prompt=session.current_prompt,
                session_id=session.id
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if result:
                self.logger.info(
                    "Moderator analysis completed - prompt change needed",
                    **log_performance("moderator_analysis", duration_ms, "ModeratorPipeline"),
                    session_id=session.id,
                    trigger=result.trigger_reason,
                    confidence=result.confidence
                )
            else:
                self.logger.debug(
                    "Moderator analysis completed - no prompt change",
                    **log_performance("moderator_analysis", duration_ms, "ModeratorPipeline"),
                    session_id=session.id
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to analyze conversation",
                **log_error(e, "analyze_conversation", "ModeratorPipeline"),
                session_id=session.id
            )
            return None


class WebSocketHandler(LoggerMixin):
    """
    Главный WebSocket обработчик для Voice2Voice системы
    Координирует все компоненты и управляет жизненным циклом соединений
    """
    
    def __init__(self, services: Dict[str, Any]):
 
        
        # Debug: проверяем доступные сервисы
        available_services = list(services.keys())
        self.logger.debug("WebSocketHandler initializing", services=available_services)
        
        # Основные сервисы
        self.session_manager = services["session_manager"]
        self.connection_manager = ConnectionManager()
        
        # Добавляем connection_manager в сервисы
        services["connection_manager"] = self.connection_manager
        
        # Voice Activity Detector для мгновенного отклика
        self.vad = VoiceActivityDetector(
            energy_threshold=0.4,   # Увеличенный порог для WebM энтропии (0.4 = 40% уникальных байт)
            speech_start_chunks=3,  # 3 чанка для подтверждения речи (~300мс) - более надежно
            silence_chunks=15       # 15 чанков тишины для конца речи (~1.5с)
        )
        
        # Smart Start Manager для оптимизации отклика
        self.smart_start_manager = SmartStartManager(
            word_threshold=1,  # Активация после 1 значимого слова
            enable_thinking_indicator=True,
            enable_tts_prewarm=True,
            min_word_length=3,  # Минимум 3 буквы (защита от "а", "и", "но")
            min_trigger_interval=2.0,  # Минимум 2 сек между срабатываниями
            confidence_threshold=0.8  # Минимум 80% уверенности STT
        )
        
        # Добавляем в сервисы
        services["smart_start_manager"] = self.smart_start_manager
        services["vad"] = self.vad
        
        # Пайплайны
        self.logger.debug("Creating AudioPipeline", services=list(services.keys()))
        self.audio_pipeline = AudioPipeline(services)
        self.logger.debug("Creating ModeratorPipeline")
        self.moderator_pipeline = ModeratorPipeline(services)
        
        # Отслеживание активных задач
        self.active_tasks: Dict[str, Set[asyncio.Task]] = {}
        self._tasks_lock = asyncio.Lock()
        
        # Получаем TTS сервисы для прогрева
        self.tts_services = {
            "elevenlabs": services.get("elevenlabs_tts"),
            "salutespeech": services.get("tts")
        }
        
        # Настраиваем callbacks для Smart Start
        self._setup_smart_start_callbacks()
    
    def _setup_smart_start_callbacks(self):
        """Настройка callbacks для Smart Start Manager"""
        
        async def send_thinking_indicator(session_id: str):
            """Отправить индикатор размышления AI"""
            try:
                await self.connection_manager.send_json(session_id, {
                    "type": "ai_thinking",
                    "status": "processing",
                    "message": "AI обрабатывает ваш запрос...",
                    "timestamp": time.time()
                })
                self.logger.debug(
                    "Smart Start: Thinking indicator sent",
                    session_id=session_id
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to send thinking indicator",
                    session_id=session_id,
                    error=str(e)
                )
        
        async def prewarm_tts(session_id: str):
            """Прогреть TTS сервисы"""
            try:
                # Прогреваем все доступные TTS сервисы параллельно
                tasks = []
                for name, service in self.tts_services.items():
                    if service and hasattr(service, 'prewarm'):
                        self.logger.debug(
                            f"Smart Start: Prewarming {name} TTS",
                            session_id=session_id
                        )
                        tasks.append(service.prewarm())
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    self.logger.debug(
                        "Smart Start: TTS prewarm completed",
                        session_id=session_id,
                        results=results
                    )
                    
            except Exception as e:
                self.logger.warning(
                    "Failed to prewarm TTS",
                    session_id=session_id,
                    error=str(e)
                )
        
        # Устанавливаем callbacks для Smart Start
        self.smart_start_manager.set_thinking_indicator_callback(send_thinking_indicator)
        self.smart_start_manager.set_tts_prewarm_callback(prewarm_tts)
        
        # Устанавливаем callback для VAD - тот же thinking indicator
        self.vad.set_speech_started_callback(send_thinking_indicator)
        
        self.logger.info("Smart Start and VAD callbacks configured")
    
    async def handle_websocket(self, websocket: WebSocket) -> None:
        """
        Главный обработчик WebSocket соединения
        
        Args:
            websocket: WebSocket соединение
        """
        session_id = None
        connection_id = None
        
        try:
            # Создаем новую сессию
            session = await self.session_manager.create_session()
            session_id = session.id
            
            # Подключаем WebSocket
            connection_id = await self.connection_manager.connect(
                websocket, 
                session_id
            )
            
            self.logger.info(
                "WebSocket session started",
                session_id=session_id,
                connection_id=connection_id
            )
            
            # Отправляем приветственное сообщение
            await self.connection_manager.send_json(session_id, {
                "type": "session_started",
                "session_id": session_id,
                "message": "Добро пожаловать! Начинайте говорить."
            })
            
            
            # Запускаем пайплайны
            await self._start_session_pipelines(websocket, session_id)
            
        except WebSocketDisconnect:
            self.logger.info(
                "WebSocket disconnected normally",
                session_id=session_id,
                connection_id=connection_id
            )
        except Exception as e:
            self.logger.error(
                "WebSocket handler error",
                **log_error(e, "websocket_handler", "WebSocketHandler"),
                session_id=session_id,
                connection_id=connection_id
            )
        finally:
            # Smart Start: очищаем данные сессии
            if hasattr(self, 'smart_start_manager') and session_id:
                await self.smart_start_manager.cleanup_session(session_id)
            
            # VAD: очищаем данные сессии
            if hasattr(self, 'vad') and session_id:
                await self.vad.cleanup_session(session_id)
            
            # Очистка ресурсов
            await self._cleanup_session(session_id, connection_id)
    
    async def _change_server_log_level(self, new_level: str, session_id: str) -> bool:
        """
        Изменение уровня логирования сервера в runtime
        
        Args:
            new_level: Новый уровень (DEBUG, INFO, WARNING, ERROR)
            session_id: ID сессии
            
        Returns:
            bool: Успешность смены
        """
        try:
            import logging
            
            # Преобразуем строку в числовой уровень
            level_mapping = {
                "DEBUG": logging.DEBUG,     # 10
                "INFO": logging.INFO,       # 20
                "WARNING": logging.WARNING, # 30
                "ERROR": logging.ERROR       # 40
            }
            
            if new_level.upper() not in level_mapping:
                self.logger.warning(
                    "Invalid log level requested",
                    session_id=session_id,
                    requested_level=new_level,
                    valid_levels=list(level_mapping.keys())
                )
                return False
            
            numeric_level = level_mapping[new_level.upper()]
            
            # Меняем уровень для root logger
            root_logger = logging.getLogger()
            old_level = root_logger.level
            root_logger.setLevel(numeric_level)
            
            # Меняем уровень для всех handlers
            for handler in root_logger.handlers:
                handler.setLevel(numeric_level)
                
            
            self.logger.info(
                "Server log level changed dynamically",
                session_id=session_id,
                old_level=old_level,
                new_level=numeric_level,
                new_level_name=new_level.upper(),
                changed_by="web_interface"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to change server log level",
                **log_error(e, "change_log_level", "WebSocketHandler"),
                session_id=session_id,
                requested_level=new_level
            )
            return False
    
    async def _start_session_pipelines(
        self, 
        websocket: WebSocket, 
        session_id: str
    ) -> None:
        """
        Запуск всех пайплайнов для сессии
        
        Args:
            websocket: WebSocket соединение
            session_id: ID сессии
        """
        async with self._tasks_lock:
            self.active_tasks[session_id] = set()
        
        try:
            # КРИТИЧЕСКИ ВАЖНО: Добавляем задачу keepalive для предотвращения таймаутов
            async def keepalive_task():
                """Периодическая отправка ping для поддержания соединения"""
                try:
                    while True:
                        await asyncio.sleep(45)  # Каждые 45 секунд (heartbeat для активности)
                        
                        # Проверяем активность сессии
                        session = await self.session_manager.get_session(session_id)
                        if not session or not session.is_active():
                            break
                        
                        # Отправляем heartbeat (НЕ ping!) для проверки активности
                        success = await self.connection_manager.send_json(session_id, {
                            "type": "heartbeat",
                            "timestamp": time.time(),
                            "session_active": True
                        })
                        
                        if not success:
                            self.logger.warning(
                                "Failed to send heartbeat",
                                session_id=session_id
                            )
                            break
                            
                        self.logger.debug(
                            "Heartbeat sent",
                            session_id=session_id
                        )
                        
                except asyncio.CancelledError:
                    self.logger.debug("Keepalive task cancelled", session_id=session_id)
                except Exception as e:
                    self.logger.error(
                        "Keepalive task error",
                        **log_error(e, "keepalive", "WebSocketHandler"),
                        session_id=session_id
                    )
            
            # Создаем генератор аудио потока с обработкой разных типов сообщений
            async def audio_stream_generator():
                audio_chunk_count = 0
                try:
                    while True:
                        # КРИТИЧНО: Проверяем активность сессии перед получением сообщения
                        session = await self.session_manager.get_session(session_id)
                        if not session or not session.is_active():
                            self.logger.info(
                                "Session inactive, stopping audio stream",
                                session_id=session_id
                            )
                            self.logger.warning(
                                "⚠️ BREAKING audio_stream_generator - session inactive!"
                            )
                            break
                            
                        # Получаем сообщение любого типа
                        message = await websocket.receive()
                        
                        # Проверяем тип сообщения
                        if "bytes" in message:
                            # Бинарные данные - аудио
                            audio_chunk_count += 1
                            chunk_size = len(message["bytes"])
                            
                            # VAD: ВРЕМЕННО ОТКЛЮЧЕН из-за ложных срабатываний на WebM тишину
                            # Проблема: WebM кодирует тишину с достаточной энтропией
                            # TODO: Реализовать декодирование WebM для анализа амплитуды
                            if False and hasattr(self, 'vad'):  # Отключено
                                try:
                                    # Обрабатываем через VAD для раннего определения речи
                                    speech_detected = await self.vad.process_audio_chunk(
                                        session_id, 
                                        message["bytes"]
                                    )
                                    if speech_detected:
                                        self.logger.info(
                                            "🎤 VAD: Speech detected immediately!",
                                            session_id=session_id,
                                            chunk_number=audio_chunk_count
                                        )
                                except Exception as e:
                                    # Не прерываем основной поток при ошибке VAD
                                    self.logger.debug(
                                        "VAD processing error (non-critical)",
                                        error=str(e),
                                        session_id=session_id
                                    )
                            
                            if audio_chunk_count % 10 == 1:  # Логируем каждый 10-й чанк
                                self.logger.info(
                                    f"📥 Received audio chunk #{audio_chunk_count} from client",
                                    session_id=session_id,
                                    chunk_size=chunk_size
                                )
                            yield message["bytes"]
                        elif "text" in message:
                            # Текстовое сообщение - JSON команда
                            try:
                                json_data = json.loads(message["text"])
                                self.logger.info(
                                    "Received JSON message",
                                    session_id=session_id,
                                    message_type=json_data.get("type", "unknown")
                                )
                                
                                # Обрабатываем команды
                                if json_data.get("type") == "ping":
                                    # Keep-alive ping
                                    await self.connection_manager.send_json(session_id, {
                                        "type": "pong",
                                        "timestamp": time.time()
                                    })
                                elif json_data.get("type") == "test":
                                    await self.connection_manager.send_json(session_id, {
                                        "type": "test_response",
                                        "message": "Тестовое сообщение получено",
                                        "timestamp": time.time()
                                    })
                                elif json_data.get("type") == "voice_mode_started":
                                    await self.connection_manager.send_json(session_id, {
                                        "type": "info",
                                        "message": "🎤 PRODUCTION Voice2Voice режим активен (Whisper + ElevenLabs)",
                                        "services": {
                                            "stt": "WhisperSTTService",
                                            "tts": "ElevenLabsTTSService"
                                        }
                                    })
                                elif json_data.get("type") == "change_log_level":
                                    # Команда смены уровня логирования от фронтенда
                                    new_level = json_data.get("level", "INFO")
                                    success = await self._change_server_log_level(new_level, session_id)
                                    
                                    await self.connection_manager.send_json(session_id, {
                                        "type": "log_level_changed",
                                        "level": new_level,
                                        "success": success,
                                        "message": f"Уровень логирования сервера изменен на {new_level}" if success else "Ошибка смены уровня логирования"
                                    })
                                    
                            except json.JSONDecodeError:
                                self.logger.warning(
                                    "Invalid JSON received",
                                    session_id=session_id,
                                    text=message["text"][:100]
                                )
                                
                except WebSocketDisconnect:
                    self.logger.warning(
                        "⚠️ CRITICAL: Audio stream ended - WebSocket disconnected",
                        session_id=session_id,
                        total_chunks=audio_chunk_count
                    )
                    return
                except Exception as e:
                    self.logger.error(
                        "⚠️ CRITICAL: Error in audio stream - This will stop transcription!",
                        **log_error(e, "audio_stream", "WebSocketHandler"),
                        session_id=session_id
                    )
                    return
                    
            # КРИТИЧНО: Если мы здесь - значит цикл while завершился нормально
            self.logger.error(
                       "🔴 CRITICAL: audio_stream_generator exited main loop - This will STOP all on!",
                       session_id=session_id
            )
            
        
            tasks = [
                asyncio.create_task(
                    self.audio_pipeline.process_audio_stream(
                        session_id,
                        audio_stream_generator()
                    )
                ),
                asyncio.create_task(
                    self.moderator_pipeline.run_moderator_for_session(session_id)
                ),
                asyncio.create_task(keepalive_task())  # Добавляем keepalive задачу
            ]
            
            async with self._tasks_lock:
                for task in tasks:
                    self.active_tasks[session_id].add(task)
                    await self.session_manager.register_session_task(session_id, task)
       
            # КРИТИЧНО: Ждем пока WebSocket активен, а не завершения первой задачи
            try:
                # Мониторим задачи пока соединение активно
                while websocket.client_state.value <= 2:  # CONNECTING=0, CONNECTED=1, CLOSING=2
         
                    failed_tasks = []
                    for task in tasks:
                        if task.done():
                            try:
                        # Проверяем не было ли исключения
                                exc = task.exception()
                                if exc:
                                    self.logger.error(
                                        f"Task failed with exception: {exc}",
                                        task_name=task.get_name(),
                                        session_id=session_id
                                    )
                                    failed_tasks.append(task)
                            except asyncio.CancelledError:
                                pass
       
                    # Если критические задачи упали - выходим
                    if failed_tasks:
                        self.logger.error(
                            f"Critical tasks failed, stopping session",
                            session_id=session_id,
                            failed_count=len(failed_tasks)
                        )
                        break
       
                    # Небольшая пауза для проверки
                    await asyncio.sleep(0.5)    

            except Exception as e:
                self.logger.error(
                    "Error in task monitoring",
                    error=str(e),
                    session_id=session_id
                )

            for task in tasks:
                if not task.done():
                    task.cancel()
                
       
            if tasks:
                await asyncio.wait(tasks, timeout=5.0)
        except Exception as e:
            self.logger.error(
                "Error starting session pipelines",
                **log_error(e, "start_pipelines", "WebSocketHandler"),
                session_id=session_id
            )
            raise
    
    async def _cleanup_session(
        self, 
        session_id: Optional[str], 
        connection_id: Optional[str]
    ) -> None:
        """
        Очистка ресурсов сессии
        
        Args:
            session_id: ID сессии
            connection_id: ID соединения
        """
        try:
            # Отключаем WebSocket
            if connection_id:
                disconnected_session_id = await self.connection_manager.disconnect(
                    connection_id
                )
                if not session_id and disconnected_session_id:
                    session_id = disconnected_session_id
            
            if session_id:
                # Отменяем активные задачи
                async with self._tasks_lock:
                    if session_id in self.active_tasks:
                        tasks = self.active_tasks[session_id]
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        
                        # Ждем завершения задач
                        if tasks:
                            await asyncio.wait(tasks, timeout=3.0)
                        
                        del self.active_tasks[session_id]
                
                # Завершаем сессию
                await self.session_manager.end_session(
                    session_id, 
                    reason="websocket_disconnect"
                )
                
                self.logger.info(
                    "Session cleanup completed",
                    session_id=session_id,
                    connection_id=connection_id
                )
            
        except Exception as e:
            self.logger.error(
                "Error during session cleanup",
                **log_error(e, "cleanup_session", "WebSocketHandler"),
                session_id=session_id,
                connection_id=connection_id
            )
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики соединений
        
        Returns:
            Dict: Статистика соединений
        """
        return {
            "active_connections": self.connection_manager.get_connection_count(),
            "active_sessions": self.session_manager.get_active_sessions_count(),
            "max_sessions": settings.max_concurrent_sessions,
            "active_tasks": sum(len(tasks) for tasks in self.active_tasks.values())
        }
    
    async def health_check(self) -> bool:
        """
        Проверка здоровья WebSocket Handler
        
        Returns:
            bool: Статус здоровья
        """
        try:
            # Проверяем компоненты
            session_manager_healthy = await self.session_manager.health_check()
            
            # Проверяем количество соединений
            connection_count = self.connection_manager.get_connection_count()
            connections_healthy = connection_count <= settings.max_concurrent_sessions
            
            return session_manager_healthy and connections_healthy
            
        except Exception as e:
            self.logger.error(
                "WebSocket handler health check failed",
                **log_error(e, "health_check", "WebSocketHandler")
            )
            return False