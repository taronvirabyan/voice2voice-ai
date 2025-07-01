"""
Оптимизированный Gemini сервис для надежной работы
Использует только проверенные рабочие модели с одним API ключом
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
import google.generativeai as genai

from ..core.config import settings
from ..core.models import TranscriptSegment, MessageRole
from ..core.exceptions import ValidationError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance


class OptimizedGeminiService(LoggerMixin):
    """
    Унифицированный сервис для работы с Gemini API
    Объединяет функции Voice AI и Moderator для максимальной эффективности
    """
    
    # Список ТОЛЬКО РАБОТАЮЩИХ моделей (проверено 30.06.2025)
    WORKING_MODELS = [
        "gemini-2.0-flash-exp",      # ✅ Самая новая и быстрая
        "gemini-2.0-flash",          # ✅ Стабильная версия 2.0
        "gemini-2.0-flash-lite",     # ✅ Облегченная версия 2.0
        "gemini-1.5-flash",          # ✅ Проверенная версия 1.5
        "gemini-1.5-flash-8b",       # ✅ Компактная модель
    ]
    
    def __init__(self):
        # Настройка Gemini API с единым ключом
        genai.configure(api_key=settings.gemini_api_key)
        
        # Используем первую работающую модель
        self.current_model_index = 0
        self.model_name = self.WORKING_MODELS[self.current_model_index]
        
        # Настройки генерации для диалога
        self.generation_config = genai.types.GenerationConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            top_p=0.8,
            top_k=40
        )
        
        # Настройки генерации для модерации (более точные)
        self.moderation_config = genai.types.GenerationConfig(
            temperature=0.3,  # Низкая температура для стабильности
            max_output_tokens=300,
            top_p=0.1,
            top_k=10
        )
        
        # Настройки безопасности (разрешаем все для корректной работы)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Создаем модели
        self._create_models()
        
        # Статистика использования
        self.request_count = 0
        self.success_count = 0
        self.model_switches = 0
        self.last_switch_time = None
        
        self.logger.info(
            "✅ Optimized Gemini Service initialized",
            model=self.model_name,
            available_models=len(self.WORKING_MODELS)
        )
    
    def _create_models(self):
        """Создание моделей для диалога и модерации"""
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
    
    async def _switch_to_next_model(self) -> bool:
        """Переключение на следующую доступную модель"""
        if self.current_model_index >= len(self.WORKING_MODELS) - 1:
            self.logger.error("All models exhausted!")
            return False
        
        self.current_model_index += 1
        self.model_name = self.WORKING_MODELS[self.current_model_index]
        self.model_switches += 1
        self.last_switch_time = time.time()
        
        self.logger.info(
            f"🔄 Switching to next model: {self.model_name}",
            model_index=self.current_model_index,
            total_switches=self.model_switches
        )
        
        # Пересоздаем модели с новым именем
        self._create_models()
        return True
    
    async def _execute_with_fallback(
        self,
        model: Any,
        prompt: str,
        operation_name: str,
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """Выполнение запроса с автоматическим переключением моделей"""
        start_time = time.time()
        self.request_count += 1
        
        while self.current_model_index < len(self.WORKING_MODELS):
            try:
                self.logger.debug(
                    f"Attempting {operation_name} with {self.model_name}",
                    session_id=session_id
                )
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )
                
                if response and response.text:
                    duration_ms = (time.time() - start_time) * 1000
                    self.success_count += 1
                    
                    self.logger.info(
                        f"✅ {operation_name} successful",
                        model=self.model_name,
                        duration_ms=duration_ms,
                        session_id=session_id
                    )
                    
                    return response.text.strip()
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Проверяем, если это ошибка квоты
                if any(indicator in error_msg for indicator in ["quota", "limit", "429", "resourceexhausted"]):
                    self.logger.warning(
                        f"Model {self.model_name} quota exceeded",
                        error=str(e),
                        session_id=session_id
                    )
                    
                    # Пытаемся переключиться на следующую модель
                    if not await self._switch_to_next_model():
                        break
                    
                    # Обновляем модель для повторной попытки
                    if operation_name.startswith("dialogue"):
                        model = self.dialogue_model
                    else:
                        model = self.moderation_model
                    
                    continue
                else:
                    # Для других ошибок - логируем и возвращаем None
                    self.logger.error(
                        f"{operation_name} error",
                        error=str(e),
                        model=self.model_name,
                        session_id=session_id
                    )
                    return None
        
        # Если все модели исчерпаны
        self.logger.error(
            f"All models exhausted for {operation_name}",
            total_attempts=self.current_model_index + 1,
            session_id=session_id
        )
        return None
    
    async def generate_dialogue_response(
        self,
        user_message: str,
        current_prompt: str,
        conversation_history: List[TranscriptSegment],
        session_id: Optional[str] = None
    ) -> str:
        """Генерация ответа для диалога"""
        
        # Построение контекста
        context = self._build_dialogue_context(
            user_message, current_prompt, conversation_history
        )
        
        # Выполнение с fallback
        response = await self._execute_with_fallback(
            self.dialogue_model,
            context,
            "dialogue_generation",
            session_id
        )
        
        if response:
            return self._postprocess_dialogue_response(response)
        else:
            return "Извините, временные технические неполадки. Попробуйте еще раз."
    
    async def analyze_for_prompt_change(
        self,
        transcript_history: List[TranscriptSegment],
        current_prompt: str,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Анализ диалога для смены промпта"""
        
        # Извлекаем текст для анализа
        conversation_text = self._extract_conversation_text(transcript_history)
        if not conversation_text:
            return None
        
        # Сначала пробуем простую проверку ключевых слов
        keyword_result = self._check_keywords(conversation_text, current_prompt)
        if keyword_result:
            return keyword_result
        
        # Если нет явных ключевых слов, используем AI анализ
        analysis_prompt = self._build_moderation_prompt(conversation_text, current_prompt)
        
        response = await self._execute_with_fallback(
            self.moderation_model,
            analysis_prompt,
            "moderation_analysis",
            session_id
        )
        
        if response:
            return self._parse_moderation_response(response)
        
        return None
    
    def _build_dialogue_context(
        self,
        user_message: str,
        current_prompt: str,
        conversation_history: List[TranscriptSegment]
    ) -> str:
        """Построение контекста для диалога"""
        context = f"""Ты - дружелюбный собеседник в голосовом диалоге на русском языке.

ТЕКУЩАЯ ЗАДАЧА: {current_prompt}

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском языке
2. Используй разговорный стиль
3. Ответ должен быть коротким (1-2 предложения)
4. Следуй текущей задаче естественно
5. Будь искренне заинтересован в собеседнике

История диалога:"""
        
        # Добавляем последние сообщения
        recent = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        for segment in recent:
            speaker = "Ты" if segment.speaker == MessageRole.ASSISTANT else "Собеседник"
            context += f"\n{speaker}: {segment.text}"
        
        context += f"\nСобеседник: {user_message}\nТы:"
        
        return context
    
    def _build_moderation_prompt(self, conversation_text: str, current_prompt: str) -> str:
        """Построение промпта для анализа"""
        return f"""Проанализируй диалог и определи, нужно ли сменить стратегию.

ТЕКУЩАЯ СТРАТЕГИЯ: {current_prompt}

ДИАЛОГ:
{conversation_text}

ПРАВИЛА:
1. Если упомянута собака → сменить на промпт про товары для собак
2. Если упомянуты финансовые проблемы → сменить на промпт про заработок
3. Если упомянуты дети → сменить на промпт про детские товары
4. Если другие интересы → сменить на универсальный продающий промпт

Ответь JSON:
{{
    "change_prompt": true/false,
    "new_prompt_type": "dog"|"money"|"children"|"sales"|null,
    "confidence": 0.0-1.0
}}"""
    
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
    
    def _check_keywords(self, text: str, current_prompt: str) -> Optional[Dict[str, Any]]:
        """Простая проверка ключевых слов"""
        text_lower = text.lower()
        
        # Ключевые слова и соответствующие промпты
        keyword_rules = {
            "dog": {
                "keywords": ["собак", "щенок", "пес", "псин"],
                "prompt": settings.dog_prompt,
                "reason": "Клиент упомянул собаку"
            },
            "money": {
                "keywords": ["денег нет", "без денег", "финансы плохо"],
                "prompt": settings.money_prompt,
                "reason": "Клиент упомянул финансовые проблемы"
            },
            "children": {
                "keywords": ["дети", "ребенок", "сын", "дочь"],
                "prompt": settings.children_prompt,
                "reason": "Клиент упомянул детей"
            }
        }
        
        for rule_type, rule in keyword_rules.items():
            for keyword in rule["keywords"]:
                if keyword in text_lower and current_prompt != rule["prompt"]:
                    return {
                        "change_prompt": True,
                        "new_prompt": rule["prompt"],
                        "trigger_keywords": [keyword],
                        "trigger_reason": rule["reason"],
                        "confidence": 0.95
                    }
        
        return None
    
    def _parse_moderation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Парсинг ответа модерации"""
        try:
            import json
            
            # Очищаем от markdown
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            result = json.loads(response.strip())
            
            if result.get("change_prompt") and result.get("confidence", 0) > 0.8:
                # Мапим тип промпта на реальный промпт
                prompt_map = {
                    "dog": settings.dog_prompt,
                    "money": settings.money_prompt,
                    "children": settings.children_prompt,
                    "sales": settings.sales_prompt
                }
                
                prompt_type = result.get("new_prompt_type")
                if prompt_type and prompt_type in prompt_map:
                    return {
                        "change_prompt": True,
                        "new_prompt": prompt_map[prompt_type],
                        "trigger_reason": f"AI detected {prompt_type} context",
                        "confidence": result.get("confidence", 0.85)
                    }
            
        except Exception as e:
            self.logger.debug(f"Failed to parse moderation response: {e}")
        
        return None
    
    def _postprocess_dialogue_response(self, text: str) -> str:
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
    
    async def health_check(self) -> bool:
        """Проверка доступности сервиса"""
        try:
            test_response = await self._execute_with_fallback(
                self.dialogue_model,
                "Скажи 'тест' на русском языке",
                "health_check"
            )
            
            return test_response is not None and "тест" in test_response.lower()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики использования"""
        return {
            "current_model": self.model_name,
            "model_index": self.current_model_index,
            "available_models": len(self.WORKING_MODELS),
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "success_rate": self.success_count / self.request_count if self.request_count > 0 else 0,
            "model_switches": self.model_switches,
            "last_switch_time": self.last_switch_time
        }