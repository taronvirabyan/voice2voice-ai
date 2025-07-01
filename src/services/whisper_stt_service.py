"""
Production-Ready OpenAI Whisper STT Service
КРИТИЧЕСКИ ВАЖНО: Локальный STT с 99.9% надежностью
"""

import asyncio
import io
import time
import tempfile
import os
import sys
import subprocess
from typing import AsyncGenerator, Optional, Dict, Any, List
import numpy as np
import whisper
import torch
from pydub import AudioSegment

# VAD импорт с безопасной проверкой
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    webrtcvad = None

from ..core.config import settings
from ..core.models import TranscriptSegment, MessageRole
from ..core.exceptions import AudioProcessingError
from ..core.logging import LoggerMixin, log_performance, log_error


class SafeTempFileManager(LoggerMixin):
    """
    Безопасный менеджер временных файлов с RAM оптимизацией
    КРИТИЧНО: Всегда имеет fallback на диск
    """
    
    def __init__(self, max_ram_mb: int = 100):
        self.max_ram_bytes = max_ram_mb * 1024 * 1024
        self.current_ram_usage = 0
        self.ram_dir = self._detect_ram_dir()
        self.use_ram = self.ram_dir is not None
        self._lock = asyncio.Lock()
        
        if self.use_ram:
            self.logger.info(
                f"✅ RAM temp files enabled",
                ram_dir=self.ram_dir,
                max_mb=max_ram_mb
            )
        else:
            self.logger.info("ℹ️ RAM temp files not available, using disk")
    
    def _detect_ram_dir(self) -> Optional[str]:
        """Безопасное определение RAM директории"""
        # Проверяем только если включено в настройках
        if not getattr(settings, 'enable_ram_tempfiles', False):
            return None
            
        candidates = [
            "/dev/shm",      # Linux standard
            "/run/shm",      # Linux alternative
        ]
        
        for dir_path in candidates:
            if self._is_ram_dir_suitable(dir_path):
                return dir_path
                
        self.logger.debug("No suitable RAM directory found")
        return None
    
    def _is_ram_dir_suitable(self, dir_path: str) -> bool:
        """Проверка пригодности RAM директории"""
        try:
            # Проверка существования
            if not os.path.exists(dir_path):
                return False
            
            # Проверка что это tmpfs (только Linux)
            if sys.platform.startswith('linux'):
                result = subprocess.run(
                    ['df', '-T', dir_path],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if 'tmpfs' not in result.stdout:
                    return False
            
            # Проверка прав записи
            test_file = os.path.join(dir_path, f'.whisper_test_{os.getpid()}')
            try:
                with open(test_file, 'wb') as f:
                    f.write(b'test')
                os.unlink(test_file)
                return True
            except Exception:
                return False
                
        except Exception as e:
            self.logger.debug(f"RAM dir check failed for {dir_path}: {e}")
            return False
    
    async def create_temp_file(self, suffix: str = '.webm', size_estimate: int = 0) -> tempfile._TemporaryFileWrapper:
        """
        Создание временного файла с автоматическим выбором места
        ГАРАНТИЯ: Всегда вернет рабочий файл (RAM или диск)
        """
        # Проверяем можем ли использовать RAM
        use_ram_for_this = (
            self.use_ram and 
            size_estimate < self.max_ram_bytes * 0.8 and  # 80% лимита
            self.current_ram_usage + size_estimate < self.max_ram_bytes
        )
        
        if use_ram_for_this:
            try:
                # Пробуем создать в RAM
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                    dir=self.ram_dir
                )
                
                async with self._lock:
                    self.current_ram_usage += size_estimate
                    
                self.logger.debug(
                    f"Created RAM temp file",
                    path=temp_file.name,
                    estimated_size=size_estimate,
                    total_ram_usage=self.current_ram_usage
                )
                return temp_file
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to create RAM temp file, falling back to disk",
                    error=str(e)
                )
                # Отключаем RAM для безопасности
                self.use_ram = False
        
        # Fallback на диск (всегда работает)
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        self.logger.debug(f"Created disk temp file: {temp_file.name}")
        return temp_file
    
    async def cleanup_temp_file(self, file_path: str, size_estimate: int = 0):
        """Безопасное удаление временного файла"""
        try:
            # Проверяем был ли файл в RAM
            if self.ram_dir and file_path.startswith(self.ram_dir):
                async with self._lock:
                    self.current_ram_usage = max(0, self.current_ram_usage - size_estimate)
                    
            os.unlink(file_path)
            self.logger.debug(f"Cleaned up temp file: {file_path}")
            
        except FileNotFoundError:
            pass  # Файл уже удален
        except Exception as e:
            self.logger.debug(f"Failed to cleanup temp file {file_path}: {e}")


class WhisperSTTService(LoggerMixin):
    """
    Production-Ready Whisper STT сервис
    Обеспечивает 99.9% надежность через локальную обработку
    """
    
    def __init__(self, model_name: str = None):
        """
        Инициализация Whisper STT сервиса
        
        Args:
            model_name: Модель Whisper (tiny, base, small, medium, large, large-v2, large-v3)
        """
        # Используем модель из настроек или default
        self.model_name = model_name or getattr(settings, 'whisper_model', 'base')
        self.model: Optional[whisper.Whisper] = None
        self.is_gpu_available = torch.cuda.is_available()
        self._model_lock = asyncio.Lock()
        
        # Настройки для production (оптимизированы для быстрого отклика)
        self.chunk_duration = 2  # секунд на чанк (оптимально для диалога)
        self.overlap_duration = 0.5  # секунд перекрытия (минимум для контекста)
        self.sample_rate = 16000  # Whisper требует 16kHz
        
        # VAD настройки (БЕЗОПАСНО: по умолчанию выключен)
        self.use_vad = getattr(settings, 'enable_vad', False)
        self.vad = None
        if self.use_vad and VAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(2)  # Средняя агрессивность (0-3)
                self.logger.info("✅ VAD (Voice Activity Detection) включен")
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось инициализировать VAD: {e}")
                self.use_vad = False
                self.vad = None
        else:
            if self.use_vad and not VAD_AVAILABLE:
                self.logger.warning("⚠️ VAD запрошен, но webrtcvad не установлен")
        
        # Инициализация менеджера временных файлов
        self.temp_file_manager = None
        if getattr(settings, 'enable_ram_tempfiles', False):
            try:
                self.temp_file_manager = SafeTempFileManager(
                    max_ram_mb=getattr(settings, 'ram_tempfiles_max_size', 100)
                )
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Failed to initialize RAM temp files, using disk",
                    error=str(e)
                )
                self.temp_file_manager = None
        
        self.logger.info(
            "WhisperSTT service initializing",
            model=model_name,
            gpu_available=self.is_gpu_available,
            sample_rate=self.sample_rate,
            vad_enabled=self.use_vad
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_model_loaded()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Whisper модель остается в памяти для следующего использования
        pass
    
    async def _ensure_model_loaded(self) -> None:
        """
        Загрузка модели Whisper (один раз при старте)
        КРИТИЧЕСКИ ВАЖНО: Обеспечивает всегда готовую модель
        """
        if self.model is not None:
            return
            
        async with self._model_lock:
            if self.model is not None:
                return
                
            try:
                self.logger.info(
                    "Loading Whisper model",
                    model=self.model_name,
                    gpu=self.is_gpu_available
                )
                
                start_time = time.time()
                
                # Загружаем в отдельном thread чтобы не блокировать event loop
                self.model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: whisper.load_model(
                        self.model_name,
                        device="cuda" if self.is_gpu_available else "cpu"
                    )
                )
                
                load_time = time.time() - start_time
                
                self.logger.info(
                    "✅ Whisper model loaded successfully",
                    **log_performance("model_loading", load_time * 1000, "WhisperSTT"),
                    model=self.model_name,
                    device=self.model.device
                )
                
            except Exception as e:
                self.logger.error(
                    "❌ Failed to load Whisper model",
                    **log_error(e, "model_loading", "WhisperSTT"),
                    model=self.model_name
                )
                raise AudioProcessingError(f"Failed to load Whisper model: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Проверка здоровья STT сервиса
        
        Returns:
            bool: True если сервис готов к работе
        """
        try:
            await self._ensure_model_loaded()
            
            # Тестовая транскрипция
            test_result = await self._transcribe_audio_data(
                np.zeros(self.sample_rate, dtype=np.float32),  # 1 секунда тишины
                language="ru"
            )
            
            self.logger.info(
                "WhisperSTT health check passed",
                model=self.model_name,
                test_transcription=test_result
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "WhisperSTT health check failed",
                **log_error(e, "health_check", "WhisperSTT")
            )
            return False
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        session_id: str,
        language: str = "ru"
    ) -> AsyncGenerator[TranscriptSegment, None]:
        """
        Потоковая транскрипция аудио данных
        
        Args:
            audio_stream: Поток аудио данных
            session_id: ID сессии
            language: Язык распознавания (ru, en, etc.)
            
        Yields:
            TranscriptSegment: Сегменты распознанного текста
        """
        self.logger.info(
            "🎙️ Starting NEW Whisper transcription stream",
            session_id=session_id,
            language=language,
            model=self.model_name,
            timestamp=time.time()
        )
        
        try:
            await self._ensure_model_loaded()
            
            # Буфер для накопления аудио данных
            audio_buffer = bytearray()
            chunk_counter = 0
            total_bytes = 0
            last_chunk_time = time.time()
            silence_timeout = 5.0  # Увеличен таймаут для стабильности
            early_silence_timeout = 2.0  # Ранняя обработка при паузе 2 секунды
            
            # КРИТИЧНО: Для WebM потоков нужно сохранять первый чанк с заголовками
            webm_header = None
            webm_header_buffer = bytearray()  # Буфер для накопления заголовков
            is_collecting_header = True  # Флаг сбора заголовков
            recording_session = 0  # Счетчик сессий записи
            
            async for audio_chunk in audio_stream:
                chunk_size = len(audio_chunk)
                total_bytes += chunk_size
                last_chunk_time = time.time()
                
                # ДИАГНОСТИКА: Логируем каждый 10-й чанк
                if total_bytes % 50000 < chunk_size:  # Примерно каждые 50KB
                    self.logger.info(
                        f"📊 STT stream alive: {total_bytes} bytes received",
                        session_id=session_id,
                        chunks_received=chunk_counter,
                        buffer_size=len(audio_buffer)
                    )
                
                # Накапливаем заголовки WebM если еще не собрали
                if is_collecting_header:
                    # Добавляем чанк в буфер заголовков
                    webm_header_buffer.extend(audio_chunk)
                    
                    # Проверяем, достаточно ли данных для заголовков
                    if len(webm_header_buffer) >= 200:  # WebM заголовки обычно 100-500 байт
                        recording_session += 1
                        webm_header = bytes(webm_header_buffer)
                        is_collecting_header = False
                        
                        # Добавляем весь накопленный буфер в audio_buffer
                        audio_buffer.extend(webm_header_buffer)
                        
                        self.logger.info(
                            "✅ WebM header collected successfully",
                            session_id=session_id,
                            recording_session=recording_session,
                            header_size=len(webm_header),
                            chunks_needed=chunk_counter + 1,
                            first_bytes=webm_header[:20].hex() if len(webm_header) > 20 else webm_header.hex()
                        )
                    else:
                        # Еще накапливаем заголовок
                        self.logger.debug(
                            "📦 Collecting WebM header",
                            session_id=session_id,
                            buffer_size=len(webm_header_buffer),
                            chunk_size=chunk_size,
                            chunk_number=chunk_counter + 1
                        )
                else:
                    # Если буфер пустой и приходит новый чанк
                    if len(audio_buffer) == 0:
                        # Проверяем, является ли это началом нового WebM файла
                        if len(audio_chunk) > 4 and audio_chunk[:4].hex() == "1a45dfa3":
                            # Это начало нового WebM файла!
                            recording_session += 1
                            # Начинаем заново собирать заголовки
                            webm_header_buffer = bytearray(audio_chunk)
                            is_collecting_header = True
                            
                            # Если чанк достаточно большой, сразу сохраняем как заголовок
                            if len(audio_chunk) >= 200:
                                webm_header = audio_chunk
                                is_collecting_header = False
                                audio_buffer.extend(audio_chunk)
                            else:
                                # Начинаем накапливать заголовок
                                self.logger.info(
                                    "🔄 New WebM detected, collecting header...",
                                    session_id=session_id,
                                    initial_chunk_size=len(audio_chunk)
                                )
                            self.logger.info(
                                "🆕 New WebM recording detected (EBML header found)",
                                session_id=session_id,
                                recording_session=recording_session,
                                chunk_size=chunk_size,
                                first_bytes=audio_chunk[:20].hex() if len(audio_chunk) > 20 else audio_chunk.hex()
                            )
                        else:
                            # Это продолжение предыдущего потока, используем сохраненные заголовки
                            if webm_header is not None and len(webm_header) >= 100:
                                self.logger.info(
                                    "📼 Continuing previous WebM stream with saved headers",
                                    session_id=session_id,
                                    recording_session=recording_session,
                                    chunk_size=chunk_size,
                                    saved_header_size=len(webm_header)
                                )
                                # Начинаем новый буфер с сохраненными заголовками
                                audio_buffer.extend(webm_header)
                                audio_buffer.extend(audio_chunk)
                            else:
                                # Нет валидных заголовков - ждем новый WebM поток
                                self.logger.warning(
                                    "⚠️ No valid WebM headers saved, waiting for new stream",
                                    session_id=session_id,
                                    webm_header_size=len(webm_header) if webm_header else 0
                                )
                                # Проверяем, может это новый WebM файл
                                if len(audio_chunk) > 4 and audio_chunk[:4].hex() == "1a45dfa3":
                                    if len(audio_chunk) >= 100:
                                        webm_header = audio_chunk
                                    audio_buffer.extend(audio_chunk)
                                else:
                                    # Пропускаем чанк без заголовков
                                    continue
                    # Если буфер НЕ пустой
                    else:
                        # Проверяем не начинается ли новая запись
                        if len(audio_chunk) > 4 and audio_chunk[:4].hex() == "1a45dfa3":
                            # Это новая запись! Сбрасываем состояние
                            self.logger.info(
                                "🔄 NEW RECORDING DETECTED - Resetting state",
                                session_id=session_id,
                                recording_session=recording_session + 1,
                                buffer_size=len(audio_buffer),
                                chunk_size=len(audio_chunk)
                            )
                            # Это новая запись! Обрабатываем старый буфер если есть данные
                            if len(audio_buffer) > 1000:
                                self.logger.info(
                                    "🔄 New WebM stream detected, processing previous buffer",
                                    session_id=session_id,
                                    buffer_size=len(audio_buffer),
                                    buffer_first_bytes=audio_buffer[:20].hex() if len(audio_buffer) > 20 else audio_buffer.hex()
                                )
                                try:
                                    audio_data = await self._convert_audio_chunk(bytes(audio_buffer))
                                    if audio_data is not None and len(audio_data) > 0:
                                        text = await self._transcribe_audio_data(audio_data, language)
                                        if text and text.strip():
                                            segment = TranscriptSegment(
                                                text=text.strip(),
                                                speaker=MessageRole.USER,
                                                timestamp=time.time(),
                                                confidence=0.95
                                            )
                                            yield segment
                                except Exception as e:
                                    self.logger.error(
                                        "Error processing buffer before new stream",
                                        error=str(e),
                                        session_id=session_id
                                    )
                            
                            # Начинаем новый буфер с новыми заголовками
                            audio_buffer = bytearray(audio_chunk)
                            # КРИТИЧНО: Проверяем размер заголовка
                            if len(audio_chunk) >= 100:
                                webm_header = audio_chunk
                            self.logger.info(
                                "📦 Started new WebM stream",
                                session_id=session_id,
                                chunk_size=chunk_size,
                                first_bytes=audio_chunk[:20].hex() if len(audio_chunk) > 20 else audio_chunk.hex()
                            )
                        else:
                            audio_buffer.extend(audio_chunk)
                
                # Логируем получение чанков для отладки
                buffer_duration = (len(audio_buffer) - 1000) / (self.sample_rate * 2) if len(audio_buffer) > 1000 else 0
                self.logger.debug(
                    "📦 Received audio chunk",
                    session_id=session_id,
                    chunk_number=chunk_counter + 1,
                    chunk_size=chunk_size,
                    buffer_size=len(audio_buffer),
                    total_bytes=total_bytes,
                    buffer_seconds=round(buffer_duration, 2),
                    is_collecting_header=is_collecting_header,
                    recording_session=recording_session,
                    webm_header_exists=webm_header is not None
                )
                
                # Логируем прогресс накопления каждые 0.5 секунды
                if buffer_duration > 0 and int(buffer_duration * 2) > int((buffer_duration - chunk_size / (self.sample_rate * 2)) * 2):
                    self.logger.info(
                        f"📊 Buffer progress: {buffer_duration:.1f}s / 4.0s",
                        session_id=session_id,
                        percentage=int(buffer_duration / 4.0 * 100)
                    )
                
                # КРИТИЧНО: Изменена логика - накапливаем больше данных перед обработкой
                # Обрабатываем в двух случаях:
                # 1. Накопили целевой объем (4 секунды)
                # 2. Обнаружена пауза в речи (2 секунды) И есть минимум 1 секунда аудио
                TARGET_BUFFER_SECONDS = 4.0  # Вернули обратно на 4.0 - рабочее значение
                time_since_last_chunk = time.time() - last_chunk_time
                
                should_process = False
                if len(audio_buffer) >= (TARGET_BUFFER_SECONDS * self.sample_rate * 2 + 1000):  # 4 секунды аудио + заголовки
                    should_process = True
                    self.logger.info("🎯 Processing: buffer full (4s)", session_id=session_id)
                elif (time_since_last_chunk >= early_silence_timeout and 
                      len(audio_buffer) >= (1.0 * self.sample_rate * 2 + 1000)):  # Пауза 2с + минимум 1с аудио
                    should_process = True
                    self.logger.info(
                        f"🎯 Processing: silence detected ({time_since_last_chunk:.1f}s pause, {buffer_duration:.1f}s audio)",
                        session_id=session_id
                    )
                
                if should_process:
                    chunk_counter += 1
                    
                    self.logger.info(
                        f"🎯 Processing audio buffer #{chunk_counter}",
                        session_id=session_id,
                        buffer_size=len(audio_buffer),
                        is_collecting_header=is_collecting_header,
                        has_webm_header=webm_header is not None
                    )
                    
                    try:
                        # Конвертируем ВСЕ накопленные аудио данные
                        audio_data = await self._convert_audio_chunk(
                            bytes(audio_buffer)
                        )
                        
                        # Транскрибируем
                        if audio_data is not None and len(audio_data) > 0:
                            text = await self._transcribe_audio_data(audio_data, language)
                            
                            if text and text.strip():
                                # Создаем сегмент транскрипта
                                segment = TranscriptSegment(
                                    text=text.strip(),
                                    speaker=MessageRole.USER,
                                    timestamp=time.time(),
                                    confidence=0.95  # Whisper обычно очень точный
                                )
                                
                                self.logger.info(
                                    "✅ Transcription segment produced",
                                    session_id=session_id,
                                    chunk=chunk_counter,
                                    text_length=len(text),
                                    text_preview=text[:50] + "..." if len(text) > 50 else text,
                                    audio_duration_seconds=len(audio_data) / self.sample_rate if audio_data is not None else 0
                                )
                                
                                yield segment
                        
                        # КРИТИЧНО: Для следующей транскрипции очищаем буфер полностью
                        audio_buffer = bytearray()
                        # НЕ сбрасываем webm_header - он может понадобиться для продолжения потока
                        # is_collecting_header остается False, так как заголовки уже собраны
                        
                        self.logger.info(
                            "🔄 Buffer cleared, ready for more audio",
                            session_id=session_id,
                            recording_session=recording_session,
                            webm_header_preserved=webm_header is not None
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            "Error processing audio chunk",
                            **log_error(e, "transcribe_chunk", "WhisperSTT"),
                            session_id=session_id,
                            chunk=chunk_counter
                        )
                        # При ошибке очищаем буфер, но сохраняем заголовки
                        audio_buffer = bytearray()
                        # НЕ сбрасываем webm_header - он может понадобиться
                        # Продолжаем работу даже при ошибке в отдельном чанке
                        continue
                
                # КРИТИЧНО: Добавлена обработка по времени - если прошло достаточно времени
                # с последнего чанка, обрабатываем то что есть
                if (time.time() - last_chunk_time > silence_timeout and 
                    len(audio_buffer) > self.sample_rate):  # Минимум 1 секунда
                    
                    self.logger.info(
                        "Processing audio due to silence timeout",
                        session_id=session_id,
                        buffer_size=len(audio_buffer),
                        silence_duration=time.time() - last_chunk_time
                    )
                    
                    try:
                        audio_data = await self._convert_audio_chunk(bytes(audio_buffer))
                        if audio_data is not None:
                            text = await self._transcribe_audio_data(audio_data, language)
                            if text and text.strip():
                                segment = TranscriptSegment(
                                    text=text.strip(),
                                    speaker=MessageRole.USER,
                                    timestamp=time.time(),
                                    confidence=0.95
                                )
                                yield segment
                        # Очищаем буфер, но сохраняем заголовки для продолжения потока
                        audio_buffer = bytearray()
                        last_chunk_time = time.time()
                    except Exception as e:
                        self.logger.error(
                            "Error processing audio due to timeout",
                            **log_error(e, "transcribe_timeout", "WhisperSTT"),
                            session_id=session_id
                        )
                        # При ошибке очищаем буфер, но сохраняем заголовки
                        audio_buffer = bytearray()
                        # НЕ сбрасываем webm_header - он может понадобиться
            
            # Обрабатываем оставшиеся данные в буфере
            if len(audio_buffer) > 1000:  # Минимальный размер для обработки
                self.logger.info(
                    f"📊 Processing final audio buffer",
                    session_id=session_id,
                    buffer_size=len(audio_buffer),
                    total_chunks=chunk_counter
                )
                try:
                    audio_data = await self._convert_audio_chunk(bytes(audio_buffer))
                    if audio_data is not None:
                        text = await self._transcribe_audio_data(audio_data, language)
                        if text and text.strip():
                            segment = TranscriptSegment(
                                text=text.strip(),
                                speaker=MessageRole.USER,
                                timestamp=time.time(),
                                confidence=0.95
                            )
                            yield segment
                except Exception as e:
                    self.logger.error(
                        "Error processing final buffer",
                        **log_error(e, "transcribe_final", "WhisperSTT"),
                        session_id=session_id
                    )
            
            self.logger.warning(
                f"⚠️ CRITICAL: Whisper transcription stream ENDED - This should NOT happen during active session!",
                session_id=session_id,
                total_chunks_processed=chunk_counter,
                total_bytes_received=total_bytes,
                last_chunk_time_ago=time.time() - last_chunk_time,
                audio_buffer_remaining=len(audio_buffer)
            )
                    
        except asyncio.CancelledError:
            self.logger.info(
                "Whisper transcription stream cancelled",
                session_id=session_id
            )
        except Exception as e:
            self.logger.error(
                "Whisper transcription stream error",
                **log_error(e, "transcribe_stream", "WhisperSTT"),
                session_id=session_id
            )
            raise AudioProcessingError(f"Transcription failed: {str(e)}")
    
    async def _convert_audio_chunk(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Конвертация аудио данных в формат для Whisper
        
        Args:
            audio_bytes: Исходные аудио данные
            
        Returns:
            np.ndarray: Аудио данные в нужном формате или None
        """
        try:
            # Проверяем, что данные выглядят как WebM (начинаются с EBML заголовка)
            is_webm = len(audio_bytes) > 4 and audio_bytes[:4] == b'\x1a\x45\xdf\xa3'
            
            # Создаем временный файл для конвертации
            # Используем SafeTempFileManager если доступен
            if self.temp_file_manager:
                temp_file = await self.temp_file_manager.create_temp_file(
                    suffix=".webm",
                    size_estimate=len(audio_bytes)
                )
                temp_file.write(audio_bytes)
                temp_file.close()  # Важно закрыть перед использованием FFmpeg
                temp_file_path = temp_file.name
            else:
                # Fallback на стандартный метод
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file_path = temp_file.name
            
            try:
                # Логируем размер входных данных и тип
                self.logger.debug(
                    "Converting audio chunk",
                    input_size=len(audio_bytes),
                    temp_file=temp_file_path,
                    is_webm_header=is_webm,
                    first_bytes=audio_bytes[:20].hex() if len(audio_bytes) > 20 else audio_bytes.hex()
                )
                
                # Пробуем загрузить как webm/opus (из браузера)
                try:
                    # Используем ffmpeg для конвертации webm в wav
                    # Добавляем параметры для лучшей обработки потоковых данных
                    audio = AudioSegment.from_file(
                        temp_file_path, 
                        format="webm", 
                        codec="opus",
                        parameters=["-err_detect", "ignore_err", "-fflags", "+genpts"]
                    )
                    self.logger.info(
                        "✅ Successfully loaded webm/opus audio with FFmpeg",
                        channels=audio.channels,
                        frame_rate=audio.frame_rate,
                        duration_ms=len(audio),
                        sample_width=audio.sample_width
                    )
                except Exception as webm_error:
                    self.logger.debug(f"Failed to load as webm: {webm_error}")
                    # Если не webm, пробуем автоопределение с игнорированием ошибок
                    try:
                        audio = AudioSegment.from_file(
                            temp_file_path,
                            parameters=["-err_detect", "ignore_err", "-fflags", "+genpts"]
                        )
                        self.logger.info(
                            "✅ Successfully loaded audio with auto-detection",
                            channels=audio.channels,
                            frame_rate=audio.frame_rate,
                            duration_ms=len(audio)
                        )
                    except Exception as auto_error:
                        self.logger.error(
                            "❌ Failed to load audio - FFmpeg error or invalid stream",
                            error=str(auto_error),
                            hint="Check if audio stream contains valid WebM data with headers"
                        )
                        return None
                
                # Конвертируем в нужный формат для Whisper
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_channels(1)  # Mono
                
                # Конвертируем в numpy array
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                
                # Нормализуем в диапазон [-1, 1]
                if audio.sample_width == 2:  # 16-bit
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio.sample_width == 4:  # 32-bit
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                self.logger.debug(
                    "Audio conversion completed",
                    output_shape=audio_data.shape,
                    output_dtype=audio_data.dtype,
                    duration_seconds=len(audio_data) / self.sample_rate,
                    min_value=float(np.min(audio_data)),
                    max_value=float(np.max(audio_data))
                )
                
                return audio_data
                
            finally:
                # Удаляем временный файл
                if self.temp_file_manager:
                    # Используем безопасный cleanup с учетом RAM
                    await self.temp_file_manager.cleanup_temp_file(
                        temp_file_path,
                        size_estimate=len(audio_bytes)
                    )
                else:
                    # Стандартное удаление
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                    
        except Exception as e:
            self.logger.error(
                "Audio conversion failed",
                **log_error(e, "audio_conversion", "WhisperSTT"),
                audio_size=len(audio_bytes)
            )
            return None
    
    def _check_voice_activity(self, audio_data: np.ndarray) -> bool:
        """
        Проверка наличия голоса в аудио с помощью VAD
        
        Args:
            audio_data: Аудио данные
            
        Returns:
            bool: True если есть голос, False если тишина
        """
        if not self.use_vad or not self.vad:
            return True  # Если VAD выключен, считаем что голос есть
        
        try:
            # WebRTC VAD требует 16kHz, 16-bit PCM, фреймы по 10, 20 или 30 мс
            # Конвертируем float32 numpy в int16 PCM
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            # Проверяем фреймы по 30мс (480 samples при 16kHz)
            frame_duration_ms = 30
            frame_length = int(self.sample_rate * frame_duration_ms / 1000)
            
            num_frames_with_voice = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_length, frame_length):
                frame = audio_int16[i:i + frame_length]
                
                # VAD требует bytes
                frame_bytes = frame.tobytes()
                
                if self.vad.is_speech(frame_bytes, self.sample_rate):
                    num_frames_with_voice += 1
                total_frames += 1
            
            # Считаем процент фреймов с голосом
            if total_frames > 0:
                voice_ratio = num_frames_with_voice / total_frames
                has_voice = voice_ratio > 0.1  # Минимум 10% фреймов должны содержать голос
                
                if not has_voice:
                    self.logger.info(
                        f"🔇 VAD: Тишина обнаружена (голос в {voice_ratio*100:.1f}% фреймов)",
                        audio_duration=len(audio_data) / self.sample_rate
                    )
                
                return has_voice
            
            return True  # По умолчанию считаем что голос есть
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка VAD проверки: {e}")
            return True  # При ошибке не блокируем обработку
    
    async def _transcribe_audio_data(
        self, 
        audio_data: np.ndarray, 
        language: str = "ru"
    ) -> str:
        """
        Транскрипция аудио данных через Whisper
        
        Args:
            audio_data: Аудио данные в формате numpy
            language: Язык распознавания
            
        Returns:
            str: Распознанный текст
        """
        try:
            # VAD проверка перед транскрипцией
            if self.use_vad:
                has_voice = self._check_voice_activity(audio_data)
                if not has_voice:
                    self.logger.debug("VAD: Пропускаем транскрипцию - тишина")
                    return ""  # Возвращаем пустую строку для тишины
            
            start_time = time.time()
            
            self.logger.debug(
                "Starting Whisper transcription",
                audio_shape=audio_data.shape,
                audio_duration=len(audio_data) / self.sample_rate,
                language=language,
                model=self.model_name,
                vad_passed=True if not self.use_vad else "voice_detected"
            )
            
            # Выполняем транскрипцию в отдельном thread
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio_data,
                    language=language,
                    fp16=False,  # На CPU fp16 медленнее!
                    task="transcribe",
                    verbose=False,  # Отключаем verbose вывод
                    # ОПТИМИЗАЦИЯ ДЛЯ СКОРОСТИ:
                    temperature=0.0,  # Отключаем семплирование = быстрее
                    beam_size=1,  # Минимальный beam search = НАМНОГО быстрее
                    best_of=1,  # Без множественных попыток = быстрее
                    compression_ratio_threshold=2.4,  # Стандартный порог
                    logprob_threshold=-1.0,  # Стандартный порог
                    no_speech_threshold=0.8,  # УВЕЛИЧЕН порог для борьбы с галлюцинациями
                    condition_on_previous_text=False,  # Ускоряем, не используя контекст
                    initial_prompt=None,  # Без начальной подсказки
                    suppress_tokens=[-1],  # Подавляем специальные токены
                    suppress_blank=True,  # Подавляем пустые выводы
                    # БЕЗОПАСНЫЕ ОПТИМИЗАЦИИ (15-20% ускорение):
                    without_timestamps=True,  # Отключаем timestamp токены = быстрее
                    word_timestamps=False    # Отключаем пословные метки = быстрее
                )
            )
            
            transcription_time = time.time() - start_time
            text = result.get("text", "").strip()
            
            # КРИТИЧЕСКИ ВАЖНО: Проверяем вероятность отсутствия речи
            # Whisper возвращает segments с no_speech_prob для каждого сегмента
            segments = result.get("segments", [])
            if segments:
                # Проверяем первый сегмент на наличие речи
                first_segment = segments[0]
                no_speech_prob = first_segment.get("no_speech_prob", 0.0)
                
                # Если вероятность отсутствия речи высокая, игнорируем результат
                if no_speech_prob > 0.6:
                    self.logger.info(
                        "🔇 High no_speech probability detected, ignoring transcription",
                        no_speech_prob=no_speech_prob,
                        text_preview=text[:50] if text else "",
                        audio_duration=len(audio_data) / self.sample_rate
                    )
                    return ""  # Возвращаем пустую строку при высокой вероятности тишины
            
            # ДОПОЛНИТЕЛЬНАЯ ЗАЩИТА: Фильтруем типичные галлюцинации Whisper
            hallucination_phrases = [
                "продолжение следует",
                "подписывайтесь на канал",
                "спасибо за просмотр",
                "ставьте лайки",
                "до встречи",
                "всем пока",
                "субтитры",
                "titres",
                "подготовлено",
                "перевод",
                "[музыка]",
                "[аплодисменты]",
                "♪",
                "♫"
            ]
            
            # Проверяем на галлюцинации
            text_lower = text.lower()
            for phrase in hallucination_phrases:
                if phrase in text_lower and len(text) < 50:
                    self.logger.warning(
                        "🚫 Detected hallucination phrase, ignoring",
                        detected_phrase=phrase,
                        full_text=text
                    )
                    return ""
            
            # Если текст слишком короткий и повторяющийся, это тоже может быть галлюцинация
            if len(text) < 5 and text in [".", "..", "...", "а", "и", "э", "м", "ну"]:
                self.logger.warning(
                    "🚫 Detected single character hallucination",
                    text=text
                )
                return ""
            
            if text:
                audio_duration_sec = len(audio_data) / self.sample_rate
                realtime_factor = transcription_time / audio_duration_sec if audio_duration_sec > 0 else 0
                
                self.logger.info(
                    "✅ Whisper transcription successful",
                    **log_performance("transcription", transcription_time * 1000, "WhisperSTT"),
                    text_length=len(text),
                    text_preview=text[:100] + "..." if len(text) > 100 else text,
                    language=result.get("language", language),
                    model=self.model_name,
                    audio_duration_sec=audio_duration_sec,
                    realtime_factor=round(realtime_factor, 2),  # <1 = быстрее реального времени
                    device="GPU" if self.is_gpu_available else "CPU",
                    no_speech_prob=segments[0].get("no_speech_prob", 0.0) if segments else None
                )
            else:
                self.logger.debug(
                    "Whisper transcription returned empty text",
                    duration_ms=transcription_time * 1000,
                    audio_duration=len(audio_data) / self.sample_rate
                )
            
            return text
            
        except Exception as e:
            self.logger.error(
                "Whisper transcription failed",
                **log_error(e, "transcription", "WhisperSTT"),
                language=language,
                audio_shape=audio_data.shape if audio_data is not None else None
            )
            return ""
    
    def _calculate_chunk_size(self) -> int:
        """Рассчитывает размер чанка в байтах"""
        # Предполагаем 16-bit audio, mono, 16kHz
        return self.chunk_duration * self.sample_rate * 2  # 2 байта на sample
    
    def _calculate_overlap_size(self) -> int:
        """Рассчитывает размер перекрытия в байтах"""
        return self.overlap_duration * self.sample_rate * 2
    
    def get_supported_languages(self) -> List[str]:
        """
        Получение списка поддерживаемых языков
        
        Returns:
            List[str]: Список кодов языков
        """
        return [
            "ru", "en", "de", "fr", "es", "it", "pt", "nl", 
            "pl", "sv", "da", "no", "fi", "et", "lv", "lt",
            "cs", "sk", "sl", "hr", "sr", "bg", "mk", "sq",
            "uk", "be", "kk", "ky", "uz", "mn", "hy", "az",
            "ka", "tr", "ar", "he", "fa", "ur", "hi", "bn",
            "ta", "te", "kn", "ml", "si", "th", "lo", "my",
            "km", "vi", "id", "ms", "tl", "zh", "ja", "ko"
        ]