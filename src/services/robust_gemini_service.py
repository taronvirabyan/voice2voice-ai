"""
Надежный Gemini сервис с автоматической ротацией API ключей
КРИТИЧЕСКИ ВАЖНО: Обеспечивает бесперебойную работу при превышении квот
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
import google.generativeai as genai

from ..core.config import settings
from ..core.models import TranscriptSegment, MessageRole, PromptUpdate
from ..core.exceptions import ValidationError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance
from .api_key_manager import APIKeyManager


class RobustGeminiService(LoggerMixin):
    """
    Сверхнадежный сервис Gemini с ротацией ключей и моделей
    Гарантирует работу даже при исчерпании квот
    """
    
    # Рабочие модели
    WORKING_MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]
    
    def __init__(self):
        # Собираем все доступные API ключи
        api_keys = [settings.gemini_api_key]
        
        if settings.gemini_api_key_2:
            api_keys.append(settings.gemini_api_key_2)
        
        if settings.gemini_api_key_3:
            api_keys.append(settings.gemini_api_key_3)
        
        # Инициализируем менеджер ключей
        self.key_manager = APIKeyManager(api_keys)
        
        # ВАЖНО: Настраиваем Gemini API с текущим ключом
        genai.configure(api_key=self.key_manager.get_current_key())
        
        # Текущая модель
        self.current_model_index = 0
        self.model_name = self.WORKING_MODELS[self.current_model_index]
        
        # Настройки генерации
        self.generation_config = genai.types.GenerationConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            top_p=0.8,
            top_k=40
        )
        
        self.moderation_config = genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=300,
            top_p=0.1,
            top_k=10
        )
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Создаем модели
        self._create_models()
        
        # Статистика
        self.total_requests = 0
        self.successful_requests = 0
        self.key_switches = 0
        self.model_switches = 0
        
        self.logger.info(
            "✅ Robust Gemini Service initialized",
            total_api_keys=len(api_keys),
            total_models=len(self.WORKING_MODELS),
            current_model=self.model_name
        )
    
    def _create_models(self):
        """Создание моделей с текущим API ключом"""
        self.dialogue_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.moderation_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.moderation_config,
            safety_settings=self.safety_settings
        )
    
    async def _execute_with_full_fallback(
        self,
        prompt: str,
        operation_name: str,
        is_moderation: bool = False,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """Выполнение с полным fallback через ключи и модели"""
        start_time = time.time()
        self.total_requests += 1
        
        # Внешний цикл по API ключам
        key_attempts = 0
        max_key_attempts = len(self.key_manager.api_keys)
        
        while key_attempts < max_key_attempts:
            # Внутренний цикл по моделям для текущего ключа
            model_attempts = 0
            
            while model_attempts < len(self.WORKING_MODELS):
                try:
                    current_key = self.key_manager.get_current_key()
                    model = self.moderation_model if is_moderation else self.dialogue_model
                    
                    self.logger.debug(
                        f"Attempting {operation_name}",
                        api_key=self.key_manager._mask_key(current_key),
                        model=self.model_name,
                        session_id=session_id
                    )
                    
                    # Пытаемся выполнить запрос
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt
                    )
                    
                    if response and response.text:
                        duration_ms = (time.time() - start_time) * 1000
                        self.successful_requests += 1
                        
                        self.logger.info(
                            f"✅ {operation_name} successful",
                            api_key=self.key_manager._mask_key(current_key),
                            model=self.model_name,
                            duration_ms=duration_ms,
                            attempts=key_attempts * len(self.WORKING_MODELS) + model_attempts + 1
                        )
                        
                        return response.text.strip()
                
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Проверяем тип ошибки
                    if any(indicator in error_msg for indicator in ["quota", "limit", "429", "resourceexhausted"]):
                        self.logger.warning(
                            f"Quota exceeded for model {self.model_name}",
                            api_key=self.key_manager._mask_key(current_key),
                            error=str(e)[:100]
                        )
                        
                        # Помечаем модель как исчерпанную для этого ключа
                        self.key_manager.mark_key_exhausted(current_key, self.model_name)
                        
                        # Пробуем следующую модель
                        model_attempts += 1
                        if model_attempts < len(self.WORKING_MODELS):
                            self.current_model_index = (self.current_model_index + 1) % len(self.WORKING_MODELS)
                            self.model_name = self.WORKING_MODELS[self.current_model_index]
                            self.model_switches += 1
                            self._create_models()
                            continue
                        
                    else:
                        # Неожиданная ошибка
                        self.logger.error(
                            f"{operation_name} unexpected error",
                            error=str(e),
                            model=self.model_name
                        )
                        return None
                
                break  # Выходим из цикла моделей если не quota error
            
            # Все модели для текущего ключа исчерпаны, пробуем следующий ключ
            key_attempts += 1
            
            if key_attempts < max_key_attempts:
                next_key = self.key_manager.switch_to_next_key()
                if next_key:
                    self.key_switches += 1
                    # Сбрасываем индекс моделей для нового ключа
                    self.current_model_index = 0
                    self.model_name = self.WORKING_MODELS[0]
                    self._create_models()
                else:
                    # Все ключи исчерпаны
                    break
        
        # Полный провал - все ключи и модели исчерпаны
        duration_ms = (time.time() - start_time) * 1000
        self.logger.error(
            f"All API keys and models exhausted for {operation_name}",
            total_attempts=key_attempts * len(self.WORKING_MODELS),
            duration_ms=duration_ms,
            session_id=session_id
        )
        
        # Показываем статус ключей
        status = self.key_manager.get_status()
        hours = status["time_until_reset"] // 3600
        minutes = (status["time_until_reset"] % 3600) // 60
        
        self.logger.error(
            f"⏰ Quota will reset in {hours}h {minutes}m at 07:00 UTC",
            keys_status=status
        )
        
        return None
    
    async def generate_response(
        self,
        user_message: str,
        current_prompt: str,
        conversation_history: List[TranscriptSegment],
        session_id: Optional[str] = None
    ) -> str:
        """Генерация ответа с полной отказоустойчивостью"""
        
        # Валидация
        if not user_message or not user_message.strip():
            return "Извините, не расслышал. Повторите пожалуйста."
        
        # Построение контекста
        context = self._build_dialogue_context(
            user_message, current_prompt, conversation_history
        )
        
        # Выполнение с полным fallback
        response = await self._execute_with_full_fallback(
            context,
            "dialogue_generation",
            is_moderation=False,
            session_id=session_id
        )
        
        if response:
            return self._postprocess_response(response)
        else:
            # Критическая ситуация - все ключи исчерпаны
            return "Извините, сервис временно недоступен из-за превышения лимитов. Попробуйте позже."
    
    async def analyze_conversation(
        self,
        transcript_history: List[TranscriptSegment],
        current_prompt: str,
        session_id: Optional[str] = None
    ) -> Optional[PromptUpdate]:
        """Анализ диалога с fallback на ключевые слова"""
        
        if not transcript_history or len(transcript_history) < 3:
            return None
        
        # Сначала проверяем ключевые слова (работает без API)
        conversation_text = self._extract_conversation_text(transcript_history)
        keyword_result = self._check_keywords(conversation_text, current_prompt)
        
        if keyword_result:
            self.logger.info(
                "Prompt change detected by keywords",
                trigger_keywords=keyword_result.trigger_keywords,
                session_id=session_id
            )
            return keyword_result
        
        # Если нет явных ключевых слов, пробуем AI анализ
        analysis_prompt = self._build_moderation_prompt(conversation_text, current_prompt)
        
        response = await self._execute_with_full_fallback(
            analysis_prompt,
            "moderation_analysis",
            is_moderation=True,
            session_id=session_id
        )
        
        if response:
            return self._parse_moderation_response(response, current_prompt)
        
        return None
    
    def _build_dialogue_context(
        self,
        user_message: str,
        current_prompt: str,
        conversation_history: List[TranscriptSegment]
    ) -> str:
        """Построение контекста диалога"""
        context = f"""Ты - дружелюбный собеседник в голосовом диалоге на русском языке.

ТЕКУЩАЯ ЗАДАЧА: {current_prompt}

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском языке
2. Используй разговорный стиль
3. Ответ должен быть коротким (1-2 предложения)
4. Следуй текущей задаче естественно
5. Будь искренне заинтересован

История диалога:"""
        
        recent = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        for segment in recent:
            speaker = "Ты" if segment.speaker == MessageRole.ASSISTANT else "Собеседник"
            context += f"\n{speaker}: {segment.text}"
        
        context += f"\nСобеседник: {user_message}\nТы:"
        return context
    
    def _extract_conversation_text(self, transcript_history: List[TranscriptSegment]) -> str:
        """Извлечение текста из истории"""
        if not transcript_history:
            return ""
        
        recent = transcript_history[-20:] if len(transcript_history) > 20 else transcript_history
        lines = []
        
        for segment in recent:
            speaker = "Клиент" if segment.speaker.value == "user" else "AI"
            lines.append(f"{speaker}: {segment.text}")
        
        return "\n".join(lines)
    
    def _check_keywords(self, text: str, current_prompt: str) -> Optional[PromptUpdate]:
        """Проверка ключевых слов (работает без API)"""
        text_lower = text.lower()
        
        # Правила переключения
        keyword_rules = {
            "dog": {
                "keywords": ["собак", "щенок", "пес", "псин", "лабрадор", "овчарк"],
                "prompt": settings.dog_prompt,
                "reason": "Клиент упомянул собаку"
            },
            "money": {
                "keywords": ["денег нет", "без денег", "финансы плохо", "не хватает денег"],
                "prompt": settings.money_prompt,
                "reason": "Клиент упомянул финансовые проблемы"
            },
            "children": {
                "keywords": ["дети", "ребенок", "сын", "дочь", "детск"],
                "prompt": settings.children_prompt,
                "reason": "Клиент упомянул детей"
            }
        }
        
        # Проверяем последние сообщения (более свежие = более важные)
        lines = text.split('\n')
        recent_text = '\n'.join(lines[-5:]) if len(lines) > 5 else text
        recent_lower = recent_text.lower()
        
        for rule_type, rule in keyword_rules.items():
            for keyword in rule["keywords"]:
                if keyword in recent_lower and current_prompt != rule["prompt"]:
                    return PromptUpdate(
                        session_id="",
                        old_prompt=current_prompt,
                        new_prompt=rule["prompt"],
                        trigger_keywords=[keyword],
                        trigger_reason=rule["reason"],
                        confidence=0.95,
                        timestamp=time.time()
                    )
        
        return None
    
    def _build_moderation_prompt(self, conversation_text: str, current_prompt: str) -> str:
        """Промпт для модерации"""
        return f"""Проанализируй диалог и определи нужна ли смена стратегии.

ТЕКУЩАЯ СТРАТЕГИЯ: {current_prompt[:50]}...

ПОСЛЕДНИЕ СООБЩЕНИЯ:
{conversation_text[-500:]}

ПРАВИЛА:
1. Собака → промпт про товары для собак
2. Финансовые проблемы → промпт про заработок
3. Дети → промпт про детские товары

Ответь ТОЛЬКО JSON:
{{"change": true/false, "type": "dog"|"money"|"children"|null}}"""
    
    def _parse_moderation_response(self, response: str, current_prompt: str) -> Optional[PromptUpdate]:
        """Парсинг ответа модерации"""
        try:
            import json
            
            # Очищаем от markdown
            response = response.strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            result = json.loads(response.strip())
            
            if result.get("change") and result.get("type"):
                prompt_map = {
                    "dog": (settings.dog_prompt, "Обнаружено упоминание собаки"),
                    "money": (settings.money_prompt, "Обнаружены финансовые проблемы"),
                    "children": (settings.children_prompt, "Обнаружено упоминание детей")
                }
                
                prompt_type = result.get("type")
                if prompt_type in prompt_map:
                    new_prompt, reason = prompt_map[prompt_type]
                    
                    if new_prompt != current_prompt:
                        return PromptUpdate(
                            session_id="",
                            old_prompt=current_prompt,
                            new_prompt=new_prompt,
                            trigger_keywords=[prompt_type],
                            trigger_reason=reason,
                            confidence=0.65,
                            timestamp=time.time()
                        )
        
        except Exception as e:
            self.logger.debug(f"Failed to parse moderation: {e}")
        
        return None
    
    def _postprocess_response(self, text: str) -> str:
        """Постобработка ответа"""
        # Удаляем форматирование
        text = text.replace("*", "").replace("_", "")
        text = text.replace("\n", " ").replace("\r", " ")
        
        # Убираем множественные пробелы
        while "  " in text:
            text = text.replace("  ", " ")
        
        # Ограничиваем длину
        sentences = text.split(". ")
        if len(sentences) > 2:
            text = ". ".join(sentences[:2]) + "."
        
        # Убираем префиксы
        for prefix in ["Ты: ", "AI: ", "Ответ: "]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text.strip() or "Понятно. Расскажите подробнее."
    
    async def analyze_with_fallback(
        self,
        transcript_history: List[TranscriptSegment],
        current_prompt: str,
        session_id: Optional[str] = None
    ) -> Optional[PromptUpdate]:
        """Анализ диалога через Gemini AI для смены промпта"""
        
        # Проверяем что есть достаточно сообщений (минимум 2 для контекста)
        if not transcript_history or len(transcript_history) < 2:
            return None
            
        # Убираем блокировку - позволяем многократные смены промпта
            
        # Извлекаем последние сообщения для контекста
        conversation_lines = []
        recent_history = transcript_history[-10:] if len(transcript_history) > 10 else transcript_history
        
        for segment in recent_history:
            speaker = "Клиент" if segment.speaker == MessageRole.USER else "AI"
            conversation_lines.append(f"{speaker}: {segment.text}")
        
        conversation_text = "\n".join(conversation_lines)
        
        # Промпт для AI анализа - ПОСТОЯННО анализируем необходимость обновления
        analysis_prompt = f"""Проанализируй ПОСЛЕДНЕЕ сообщение пользователя и определи, нужно ли обновить промпт для адаптации к новому контексту.

ПОСЛЕДНЕЕ СООБЩЕНИЕ: {conversation_text.split('\n')[-1] if conversation_text else ''}

ПОЛНЫЙ ДИАЛОГ:
{conversation_text}

ТЕКУЩИЙ ПРОМПТ: {"обычный диалог" if "Узнай собеседника" in current_prompt else "режим предложений"}

ЗАДАЧА: ВСЕГДА анализируй, изменился ли контекст разговора!
- Если пользователь перешёл на новую тему - ОБНОВИ промпт (activate_sales: true)
- Если контекст не изменился - оставь текущий промпт (activate_sales: false)

Примеры изменения контекста, требующие обновления:
- Переход от животных к финансам
- Переход от хобби к проблемам
- Переход от работы к отдыху
- ЛЮБАЯ смена темы разговора

Ответь JSON:
{{
    "activate_sales": true/false,
    "reason": "краткое объяснение почему",
    "detected_interests": ["текущая тема разговора"]
}}"""

        try:
            # Используем AI для анализа
            response = await self._execute_with_full_fallback(
                analysis_prompt,
                "prompt_analysis",
                is_moderation=True,
                session_id=session_id
            )
            
            if response:
                # Парсим ответ
                import json
                response_clean = response.strip()
                if "```" in response_clean:
                    response_clean = response_clean.split("```")[1]
                    if response_clean.startswith("json"):
                        response_clean = response_clean[4:]
                        
                result = json.loads(response_clean.strip())
                
                if result.get("activate_sales", False):
                    self.logger.info(
                        "🤖 AI decided to activate sales prompt",
                        reason=result.get("reason", ""),
                        interests=result.get("detected_interests", []),
                        session_id=session_id
                    )
                    
                    return PromptUpdate(
                        session_id=session_id or "",
                        old_prompt=current_prompt,
                        new_prompt=settings.sales_prompt,
                        trigger_keywords=result.get("detected_interests", []),
                        trigger_reason=f"AI анализ: {result.get('reason', 'Обнаружена возможность для продаж')}",
                        confidence=0.65,
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            self.logger.debug(f"AI analysis failed, no prompt change: {e}")
            
        return None
    
    async def health_check(self) -> bool:
        """Проверка здоровья с использованием любого доступного ключа"""
        response = await self._execute_with_full_fallback(
            "Скажи 'тест'",
            "health_check",
            is_moderation=False
        )
        
        return response is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика работы сервиса"""
        key_status = self.key_manager.get_status()
        
        return {
            "service": "RobustGeminiService",
            "current_model": self.model_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "key_switches": self.key_switches,
            "model_switches": self.model_switches,
            "api_keys": key_status
        }