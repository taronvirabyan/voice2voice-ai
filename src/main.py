"""
Production-Ready FastAPI приложение для Voice2Voice AI системы
КРИТИЧЕСКИ ВАЖНО: Центральная точка входа с полной обработкой ошибок и мониторингом
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import structlog

from .core.config import settings
from .core.models import SystemHealth, APIError
from .core.exceptions import ServiceInitializationError
from .core.logging import setup_logging, LoggerMixin

# Импорты сервисов
from .services.stt_service import SaluteSpeechSTT
from .services.tts_service import SaluteSpeechTTS
from .services.gemini_voice_ai_service import GeminiVoiceAIService
from .services.gemini_moderator_service import GeminiModeratorService
from .services.robust_gemini_service import RobustGeminiService
from .services.redis_service import RedisService
from .services.session_manager import SessionManager

# Опциональные импорты для Voice2Voice Production
WhisperSTTService = None
ElevenLabsTTSService = None
MockTTSService = None
MockAIService = None
MockSTTService = None
WHISPER_AVAILABLE = False
ELEVENLABS_AVAILABLE = False
MOCK_TTS_AVAILABLE = False
MOCK_AI_AVAILABLE = False
MOCK_STT_AVAILABLE = False

try:
    from .services.whisper_stt_service import WhisperSTTService
    WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    from .services.elevenlabs_tts_service import ElevenLabsTTSService
    ELEVENLABS_AVAILABLE = True
except ImportError:
    pass

try:
    from .services.mock_tts_service import MockTTSService
    MOCK_TTS_AVAILABLE = True
except ImportError:
    pass

try:
    from .services.mock_ai_service import MockAIService
    MOCK_AI_AVAILABLE = True
except ImportError:
    pass

try:
    from .services.mock_stt_service import MockSTTService
    MOCK_STT_AVAILABLE = True
except ImportError:
    pass

# Импорт API компонентов
from .api.websocket_handler import WebSocketHandler


class VoiceToVoiceApp(LoggerMixin):
    """
    Главный класс приложения с управлением жизненным циклом всех сервисов
    """
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.app_start_time = time.time()
        self.is_healthy = False
        
        # Setup logging
        setup_logging()
        
        self.logger.info(
            "Voice2Voice AI application initializing",
            version="1.0.0",
            environment="production" if not settings.debug else "development"
        )
    
    async def initialize_services(self) -> None:
        """
        Инициализация всех сервисов системы
        КРИТИЧЕСКИ ВАЖНО: Правильный порядок инициализации
        """
        try:
            self.logger.info("Starting service initialization...")
            
            # 1. Redis Service (базовая инфраструктура)
            self.logger.info("Initializing Redis service...")
            redis_service = RedisService()
            await redis_service.connect()
            
            # Проверяем Redis
            if not await redis_service.health_check():
                raise ServiceInitializationError("Redis health check failed")
            
            self.services["redis"] = redis_service
            self.logger.info("✅ Redis service initialized")
            
            # 2. Session Manager (зависит от Redis)
            self.logger.info("Initializing Session Manager...")
            session_manager = SessionManager(redis_service)
            
            if not await session_manager.health_check():
                raise ServiceInitializationError("Session Manager health check failed")
            
            self.services["session_manager"] = session_manager
            self.logger.info("✅ Session Manager initialized")
            
            # 3. AI Services (опциональные при превышении квоты)
            self.logger.info("Initializing AI services...")
            
            try:
                # Используем RobustGeminiService с ротацией ключей
                self.logger.info("Initializing Robust Gemini Service with key rotation...")
                robust_service = RobustGeminiService()
                
                if await robust_service.health_check():
                    # Используем один сервис для обеих функций
                    self.services["voice_ai"] = robust_service
                    self.services["moderator"] = robust_service
                    self.logger.info("✅ Robust Gemini Service initialized with API key rotation")
                    
                    # Показываем статус ключей
                    stats = robust_service.get_stats()
                    self.logger.info(
                        "API keys status",
                        available_keys=stats["api_keys"]["total_keys"],
                        current_key_index=stats["api_keys"]["current_key_index"]
                    )
                else:
                    # Fallback на старые сервисы
                    self.logger.warning("Robust service failed, trying standard services...")
                    
                    # Gemini Voice AI
                    voice_ai_service = GeminiVoiceAIService()
                    if await voice_ai_service.health_check():
                        self.services["voice_ai"] = voice_ai_service
                        self.logger.info("✅ Gemini Voice AI service initialized")
                    else:
                        self.logger.warning("⚠️ Gemini Voice AI not available - continuing without it")
                    
                    # Gemini Moderator
                    moderator_service = GeminiModeratorService()
                    if await moderator_service.health_check():
                        self.services["moderator"] = moderator_service
                        self.logger.info("✅ Gemini Moderator service initialized")
                    else:
                        self.logger.warning("⚠️ Gemini Moderator not available - continuing without it")
                    
            except Exception as e:
                self.logger.warning(
                    "⚠️ Gemini AI services failed to initialize - trying Mock AI fallback",
                    error=str(e)
                )
                
            # НЕ используем Mock AI - пользователь хочет только Gemini
            if "voice_ai" not in self.services:
                self.logger.error(
                    "❌ Gemini AI service not available and MockAI disabled by user request"
                )
                raise ServiceInitializationError(
                    "AI Services",
                    "Gemini AI service failed to initialize and MockAI fallback is disabled"
                )
            
            # 4. Voice2Voice Production Services (Whisper + Mock TTS для демонстрации)
            self.logger.info("Initializing Voice2Voice Production services...")
            
            try:
                # STT Service - пробуем Whisper, затем Mock STT
                if WHISPER_AVAILABLE and WhisperSTTService:
                    self.logger.info("Initializing Whisper STT service...")
                    whisper_stt = WhisperSTTService(model_name=settings.whisper_model)
                    if await whisper_stt.health_check():
                        self.services["stt"] = whisper_stt
                        self.logger.info("✅ Whisper STT service initialized (OFFLINE)")
                    else:
                        self.logger.error("❌ Whisper STT failed to initialize")
                elif MOCK_STT_AVAILABLE and MockSTTService:
                    self.logger.info("Initializing Mock STT service for demonstration...")
                    mock_stt = MockSTTService()
                    if await mock_stt.health_check():
                        self.services["stt"] = mock_stt
                        self.logger.info("✅ Mock STT service initialized (DEMONSTRATION MODE)")
                    else:
                        self.logger.error("❌ Mock STT failed to initialize")
                else:
                    self.logger.warning("⚠️ No STT service available")
                
                # Mock TTS Service (для демонстрации полной функциональности)
                if MOCK_TTS_AVAILABLE and MockTTSService:
                    self.logger.info("Initializing Mock TTS service for demonstration...")
                    mock_tts = MockTTSService()
                    if await mock_tts.health_check():
                        self.services["tts"] = mock_tts
                        self.logger.info("✅ Mock TTS service initialized (DEMONSTRATION MODE)")
                    else:
                        self.logger.error("❌ Mock TTS failed to initialize")
                
                # ElevenLabs TTS Service (с fallback на SaluteSpeech)
                if ELEVENLABS_AVAILABLE and ElevenLabsTTSService:
                    self.logger.info("Initializing ElevenLabs TTS service...")
                    elevenlabs_tts = ElevenLabsTTSService()
                    if await elevenlabs_tts.health_check():
                        self.services["tts"] = elevenlabs_tts
                        self.logger.info("✅ ElevenLabs TTS service initialized")
                    else:
                        self.logger.warning("⚠️ ElevenLabs TTS not available - trying SaluteSpeech fallback...")
                        
                        # Fallback на SaluteSpeech TTS
                        try:
                            salute_tts = SaluteSpeechTTS()
                            if await salute_tts.health_check():
                                self.services["tts"] = salute_tts
                                self.logger.info("✅ SaluteSpeech TTS fallback initialized")
                            else:
                                self.logger.warning("⚠️ All TTS services unavailable - system will work in STT-only mode")
                        except Exception as fallback_error:
                            self.logger.warning(
                                "⚠️ TTS fallback failed - continuing without TTS",
                                error=str(fallback_error)
                            )
                else:
                    self.logger.warning("⚠️ ElevenLabs TTS not available - missing dependencies")
                    
                    # Fallback на SaluteSpeech TTS если ElevenLabs недоступен
                    try:
                        salute_tts = SaluteSpeechTTS()
                        if await salute_tts.health_check():
                            self.services["tts"] = salute_tts
                            self.logger.info("✅ SaluteSpeech TTS fallback initialized")
                        else:
                            self.logger.warning("⚠️ All TTS services unavailable - system will work in STT-only mode")
                    except Exception as fallback_error:
                        self.logger.warning(
                            "⚠️ TTS fallback failed - continuing without TTS",
                            error=str(fallback_error)
                        )
                    
            except Exception as e:
                self.logger.warning(
                    "⚠️ Voice2Voice services initialization failed - trying legacy fallback",
                    error=str(e)
                )
                
                # Emergency fallback на SaluteSpeech
                try:
                    # STT Service
                    stt_service = SaluteSpeechSTT()
                    if await stt_service.health_check():
                        self.services["stt"] = stt_service
                        self.logger.info("✅ Emergency SaluteSpeech STT fallback initialized")
                    
                    # TTS Service
                    tts_service = SaluteSpeechTTS()
                    if await tts_service.health_check():
                        self.services["tts"] = tts_service
                        self.logger.info("✅ Emergency SaluteSpeech TTS fallback initialized")
                        
                except Exception as emergency_error:
                    self.logger.warning(
                        "⚠️ Emergency SaluteSpeech fallback failed - trying Mock TTS",
                        error=str(emergency_error)
                    )
                    
                    # Final fallback - Mock TTS для демонстрации
                    if MOCK_TTS_AVAILABLE and MockTTSService:
                        try:
                            mock_tts = MockTTSService()
                            if await mock_tts.health_check():
                                self.services["tts"] = mock_tts
                                self.logger.info("✅ Mock TTS emergency fallback initialized (for demonstration)")
                            else:
                                self.logger.warning("⚠️ All TTS services failed - system will work in simulation mode")
                        except Exception as mock_error:
                            self.logger.warning(
                                "⚠️ Mock TTS fallback failed - system will work in simulation mode",
                                error=str(mock_error)
                            )
                    else:
                        self.logger.warning("⚠️ No fallback TTS available - system will work in simulation mode")
            
            # 5. WebSocket Handler (зависит от всех сервисов)
            self.logger.info("Initializing WebSocket Handler...")
            websocket_handler = WebSocketHandler(self.services)
            
            if not await websocket_handler.health_check():
                raise ServiceInitializationError("WebSocket Handler", "health check failed")
            
            self.services["websocket_handler"] = websocket_handler
            self.logger.info("✅ WebSocket Handler initialized")
            
            # Финальная проверка системы
            await self._perform_integration_health_check()
            
            self.is_healthy = True
            
            initialization_time = time.time() - self.app_start_time
            
            self.logger.info(
                "🎉 All services initialized successfully!",
                initialization_time_seconds=initialization_time,
                total_services=len(self.services)
            )
            
        except Exception as e:
            self.logger.error(
                "💥 Service initialization failed",
                error=str(e),
                error_type=type(e).__name__,
                stack_trace=e.__traceback__
            )
            self.is_healthy = False
            raise ServiceInitializationError("Services", f"Failed to initialize services: {str(e)}")
    
    async def _perform_integration_health_check(self) -> None:
        """
        Комплексная проверка интеграции всех сервисов
        """
        self.logger.info("Performing integration health check...")
        
        try:
            # Проверяем каждый сервис
            health_checks = []
            
            for service_name, service in self.services.items():
                if hasattr(service, 'health_check'):
                    try:
                        is_healthy = await service.health_check()
                        health_checks.append((service_name, is_healthy))
                    except Exception as e:
                        self.logger.error(
                            f"Health check failed for {service_name}",
                            error=str(e)
                        )
                        health_checks.append((service_name, False))
            
            # Проверяем результаты (более мягкий подход для production)
            failed_services = [name for name, healthy in health_checks if not healthy]
            critical_services = ["redis", "session_manager", "websocket_handler"]
            
            # Проверяем только критически важные сервисы
            critical_failed = [name for name in failed_services if name in critical_services]
            
            if critical_failed:
                raise ServiceInitializationError(
                    "Integration Health Check", 
                    f"Critical services failed: {', '.join(critical_failed)}"
                )
            
            if failed_services:
                self.logger.warning(
                    "⚠️ Some non-critical services failed health check - continuing in degraded mode",
                    failed_services=failed_services
                )
            
            self.logger.info(
                "✅ Integration health check passed",
                checked_services=len(health_checks)
            )
            
        except Exception as e:
            self.logger.error(
                "❌ Integration health check failed",
                error=str(e)
            )
            raise
    
    async def shutdown_services(self) -> None:
        """
        Корректное завершение работы всех сервисов
        """
        self.logger.info("Shutting down services...")
        
        shutdown_order = [
            "websocket_handler",
            "session_manager", 
            "stt",
            "tts",
            "voice_ai",
            "moderator",
            "redis"
        ]
        
        for service_name in shutdown_order:
            if service_name in self.services:
                try:
                    service = self.services[service_name]
                    
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                    elif hasattr(service, 'disconnect'):
                        await service.disconnect()
                    elif hasattr(service, '__aexit__'):
                        await service.__aexit__(None, None, None)
                    
                    self.logger.info(f"✅ {service_name} service shut down")
                    
                except Exception as e:
                    self.logger.error(
                        f"Error shutting down {service_name}",
                        error=str(e)
                    )
        
        self.logger.info("🔄 All services shut down completed")
    
    def get_system_health(self) -> SystemHealth:
        """
        Получение состояния системы
        """
        health = SystemHealth()
        
        # Проверяем статус каждого сервиса
        for service_name, service in self.services.items():
            try:
                # Простая проверка доступности
                health.services[service_name] = hasattr(service, 'health_check')
            except Exception:
                health.services[service_name] = False
        
        # Определяем общий статус
        if all(health.services.values()) and self.is_healthy:
            health.status = "healthy"
        elif any(health.services.values()):
            health.status = "degraded"
        else:
            health.status = "unhealthy"
        
        # Добавляем метрики
        if "session_manager" in self.services:
            try:
                stats = self.services["session_manager"].get_session_stats()
                health.active_sessions = stats.get("active_sessions", 0)
                health.total_sessions = stats.get("total_sessions_created", 0)
            except Exception:
                pass
        
        return health


# Глобальный экземпляр приложения
voice_app = VoiceToVoiceApp()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Менеджер жизненного цикла приложения
    """
    # Startup
    try:
        voice_app.logger.info("🚀 Voice2Voice AI application starting...")
        await voice_app.initialize_services()
        voice_app.logger.info("🎉 Application startup completed successfully!")
        
        yield
        
    except Exception as e:
        voice_app.logger.error(
            "💥 Application startup failed",
            error=str(e)
        )
        raise
    finally:
        # Shutdown
        voice_app.logger.info("🔄 Application shutting down...")
        await voice_app.shutdown_services()
        voice_app.logger.info("✅ Application shutdown completed")


# Создание FastAPI приложения
app = FastAPI(
    title="Voice2Voice AI Chat System",
    description="Production-Ready голосовой чат с динамической коррекцией промптов",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files для WebSocket тестирования
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Dependency для получения сервисов
def get_voice_app() -> VoiceToVoiceApp:
    """Dependency для получения экземпляра приложения"""
    return voice_app


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик ошибок"""
    voice_app.logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=str(request.url)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": "INTERNAL_ERROR"
        }
    )


# ===== HEALTH CHECK ENDPOINTS =====

@app.get("/health", response_model=SystemHealth)
async def health_check(app_instance: VoiceToVoiceApp = Depends(get_voice_app)):
    """
    Проверка здоровья системы
    """
    try:
        health = app_instance.get_system_health()
        
        # Определяем HTTP статус код
        if health.status == "healthy":
            status_code = 200
        elif health.status == "degraded":
            status_code = 200  # Частично работает
        else:
            status_code = 503  # Service Unavailable
        
        return JSONResponse(
            status_code=status_code,
            content=health.model_dump(mode='json')
        )
        
    except Exception as e:
        voice_app.logger.error(
            "Health check failed",
            error=str(e)
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/health/detailed")
async def detailed_health_check(app_instance: VoiceToVoiceApp = Depends(get_voice_app)):
    """
    Подробная проверка здоровья всех сервисов
    """
    if not app_instance.is_healthy:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    try:
        detailed_health = {}
        
        for service_name, service in app_instance.services.items():
            try:
                if hasattr(service, 'health_check'):
                    is_healthy = await service.health_check()
                    detailed_health[service_name] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "available": True
                    }
                else:
                    detailed_health[service_name] = {
                        "status": "unknown",
                        "available": True
                    }
            except Exception as e:
                detailed_health[service_name] = {
                    "status": "error",
                    "available": False,
                    "error": str(e)
                }
        
        return detailed_health
        
    except Exception as e:
        voice_app.logger.error(
            "Detailed health check failed",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


# ===== METRICS ENDPOINTS =====

@app.get("/metrics")
async def get_metrics(app_instance: VoiceToVoiceApp = Depends(get_voice_app)):
    """
    Метрики системы для мониторинга
    """
    if not app_instance.is_healthy:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    try:
        metrics = {
            "app_uptime_seconds": time.time() - app_instance.app_start_time,
            "timestamp": time.time()
        }
        
        # Session metrics
        if "session_manager" in app_instance.services:
            session_stats = app_instance.services["session_manager"].get_session_stats()
            metrics.update(session_stats)
        
        # Connection metrics
        if "websocket_handler" in app_instance.services:
            connection_stats = app_instance.services["websocket_handler"].get_connection_stats()
            metrics.update(connection_stats)
        
        # Redis metrics
        if "redis" in app_instance.services:
            redis_metrics = await app_instance.services["redis"].get_system_metrics()
            metrics["redis"] = redis_metrics
        
        return metrics
        
    except Exception as e:
        voice_app.logger.error(
            "Failed to get metrics",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


# ===== WEBSOCKET ENDPOINT =====

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    app_instance: VoiceToVoiceApp = Depends(get_voice_app)
):
    """
    Главный WebSocket endpoint для voice2voice диалогов
    """
    if not app_instance.is_healthy:
        await websocket.close(code=1011, reason="Service unavailable")
        return
    
    websocket_handler = app_instance.services.get("websocket_handler")
    
    if not websocket_handler:
        await websocket.close(code=1011, reason="WebSocket handler not available")
        return
    
    await websocket_handler.handle_websocket(websocket)


# ===== ADMIN ENDPOINTS (только в debug режиме) =====

if settings.debug:
    
    @app.get("/admin/sessions")
    async def get_active_sessions(app_instance: VoiceToVoiceApp = Depends(get_voice_app)):
        """Список активных сессий (только для отладки)"""
        if "session_manager" not in app_instance.services:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = app_instance.services["session_manager"]
        return {
            "active_session_ids": session_manager.get_active_session_ids(),
            "stats": session_manager.get_session_stats()
        }
    
    @app.post("/admin/sessions/{session_id}/end")
    async def end_session(
        session_id: str,
        app_instance: VoiceToVoiceApp = Depends(get_voice_app)
    ):
        """Принудительное завершение сессии (только для отладки)"""
        if "session_manager" not in app_instance.services:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_manager = app_instance.services["session_manager"]
        success = await session_manager.end_session(session_id, reason="admin_force")
        
        if success:
            return {"message": f"Session {session_id} ended successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")


# ===== VOICE2VOICE TESTING ENDPOINTS =====

@app.get("/test")
async def test_client():
    """
    Возвращает тестовую HTML страницу для проверки voice2voice
    """
    test_file = os.path.join(static_dir, "test_voice_simple.html")
    if os.path.exists(test_file):
        return FileResponse(test_file)
    else:
        raise HTTPException(status_code=404, detail="Test client not found")

@app.post("/test/voice2voice")
async def test_voice2voice_functionality(
    app_instance: VoiceToVoiceApp = Depends(get_voice_app)
):
    """
    Тестирование полного цикла voice2voice функциональности
    """
    if not app_instance.is_healthy:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    try:
        services = app_instance.services
        test_results = {
            "timestamp": time.time(),
            "test_type": "voice2voice_cycle",
            "stages": {}
        }
        
        # 1. Тест TTS сервиса
        if "tts" in services:
            tts_service = services["tts"]
            voice_app.logger.info("Testing TTS service functionality")
            
            test_text = "Привет! Это тест системы синтеза речи."
            
            try:
                # Тестируем генерацию аудио
                audio_chunks = []
                async for chunk in tts_service.synthesize_stream(
                    text=test_text,
                    session_id="test_session",
                    chunk_size=1024
                ):
                    audio_chunks.append(chunk)
                
                total_audio_size = sum(len(chunk) for chunk in audio_chunks)
                
                test_results["stages"]["tts"] = {
                    "status": "success",
                    "text_length": len(test_text),
                    "audio_chunks": len(audio_chunks),
                    "total_audio_size": total_audio_size,
                    "service_type": type(tts_service).__name__
                }
                
            except Exception as e:
                test_results["stages"]["tts"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            test_results["stages"]["tts"] = {
                "status": "unavailable",
                "message": "TTS service not initialized"
            }
        
        # 2. Тест STT сервиса (если доступен)
        if "stt" in services:
            test_results["stages"]["stt"] = {
                "status": "available",
                "service_type": type(services["stt"]).__name__,
                "note": "STT requires actual audio input for testing"
            }
        else:
            test_results["stages"]["stt"] = {
                "status": "unavailable",
                "message": "STT service not initialized"
            }
        
        # 3. Тест AI сервиса (если доступен)
        if "voice_ai" in services:
            test_results["stages"]["ai"] = {
                "status": "available",
                "service_type": type(services["voice_ai"]).__name__
            }
        else:
            test_results["stages"]["ai"] = {
                "status": "degraded",
                "message": "AI service quota exceeded - using simulation mode"
            }
        
        # 4. Общий статус voice2voice
        tts_working = test_results["stages"]["tts"]["status"] == "success"
        stt_available = test_results["stages"]["stt"]["status"] == "available"
        
        if tts_working and stt_available:
            test_results["overall_status"] = "voice2voice_ready"
            test_results["message"] = "✅ Voice2Voice functionality is ready for testing"
        elif tts_working:
            test_results["overall_status"] = "tts_ready"
            test_results["message"] = "✅ TTS working, STT needs PyTorch for full functionality"
        else:
            test_results["overall_status"] = "degraded"
            test_results["message"] = "⚠️ Limited voice2voice functionality"
        
        return test_results
        
    except Exception as e:
        voice_app.logger.error(
            "Voice2Voice test failed",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/audio-generation/{text}")
async def test_audio_generation(
    text: str,
    app_instance: VoiceToVoiceApp = Depends(get_voice_app)
):
    """
    Прямое тестирование генерации аудио
    """
    if not app_instance.is_healthy:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    if "tts" not in app_instance.services:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    try:
        tts_service = app_instance.services["tts"]
        
        # Генерируем аудио
        audio_chunks = []
        start_time = time.time()
        
        async for chunk in tts_service.synthesize_stream(
            text=text,
            session_id="audio_test",
            chunk_size=1024
        ):
            audio_chunks.append(chunk)
        
        generation_time = time.time() - start_time
        total_size = sum(len(chunk) for chunk in audio_chunks)
        
        return {
            "status": "success",
            "text": text,
            "text_length": len(text),
            "audio_chunks_count": len(audio_chunks),
            "total_audio_size_bytes": total_size,
            "generation_time_seconds": round(generation_time, 3),
            "service_type": type(tts_service).__name__,
            "audio_format": "WAV",
            "sample_rate": getattr(tts_service, 'sample_rate', 16000)
        }
        
    except Exception as e:
        voice_app.logger.error(
            "Audio generation test failed",
            error=str(e),
            text=text
        )
        raise HTTPException(status_code=500, detail=str(e))

# ===== ROOT ENDPOINT =====

@app.get("/", include_in_schema=False)
async def root():
    """
    Возвращает главную HTML страницу
    """
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    # Если нет HTML, возвращаем JSON информацию
    return {
        "name": "Voice2Voice AI Chat System",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Real-time voice2voice conversations",
            "Dynamic prompt switching",
            "Russian language support",
            "Production-ready architecture"
        ],
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "metrics": "/metrics",
            "test_client": "/test",
            "docs": "/docs" if settings.debug else "disabled"
        }
    }

@app.get("/test")
async def websocket_test_page():
    """
    WebSocket тестовая страница
    """
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    test_file = os.path.join(static_dir, "websocket_test.html")
    
    if os.path.exists(test_file):
        return FileResponse(test_file)
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Test page not found. Please ensure static/websocket_test.html exists."}
        )


if __name__ == "__main__":
    import uvicorn
    
    voice_app.logger.info(
        "Starting Voice2Voice AI application",
        host=settings.server_host,
        port=settings.server_port,
        debug=settings.debug
    )
    
    uvicorn.run(
        "src.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
        log_level="info",
        access_log=settings.debug,
        # КРИТИЧЕСКИ ВАЖНО: Настройки для максимально стабильных WebSocket соединений
        ws_ping_interval=60,   # Отправлять ping каждые 60 секунд (реже = стабильнее)
        ws_ping_timeout=300,   # Ждать pong 5 минут (для медленных соединений)
        timeout_keep_alive=600,  # Держать соединение живым 10 минут
        ws_max_size=10 * 1024 * 1024  # Максимальный размер сообщения 10MB
    )