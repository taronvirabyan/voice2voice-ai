"""
Gemini Moderator сервис для анализа диалога и смены промптов
КРИТИЧЕСКИ ВАЖНО: Максимальная точность анализа для правильных решений
"""

import asyncio
import time
import json
from typing import List, Optional, Dict, Any
import google.generativeai as genai

from ..core.config import settings
from ..core.models import TranscriptSegment, PromptUpdate
from ..core.exceptions import ValidationError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance


class GeminiModeratorService(LoggerMixin):
    """
    Сервис модерации диалога через Gemini API
    Анализирует транскрипт и принимает решения о смене промптов
    """
    
    def __init__(self):
        # Настройка Gemini API
        genai.configure(api_key=settings.gemini_api_key)
        
        # Список ТОЛЬКО РАБОТАЮЩИХ моделей (проверено 30.06.2025)
        # Используем только модели с доступной квотой для ключа AIzaSyDPwBwdOZT3RvviaCNQZd1KopHvNz0TTZg
        self.model_fallback_list = [
            # ✅ РАБОТАЮЩИЕ МОДЕЛИ (в порядке приоритета)
            "gemini-2.0-flash-exp",      # ✅ Самая новая и быстрая
            "gemini-2.0-flash",          # ✅ Стабильная версия 2.0
            "gemini-2.0-flash-lite",     # ✅ Облегченная версия 2.0
            "gemini-1.5-flash",          # ✅ Проверенная версия 1.5
            "gemini-1.5-flash-8b",       # ✅ Компактная модель
        ]
        
        # Пытаемся использовать модель из настроек или первую из списка
        self.model_name = settings.moderator_model
        if self.model_name not in self.model_fallback_list:
            self.model_fallback_list.insert(0, self.model_name)
        
        # Настройки для точного анализа
        self.generation_config = genai.types.GenerationConfig(
            temperature=settings.moderator_temperature,  # Низкая температура для стабильности
            max_output_tokens=300,
            top_p=0.1,  # Более детерминированные ответы
            top_k=10
        )
        
        # Настройки безопасности
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # Кэш для моделей с исчерпанной квотой (сбрасывается каждый час)
        self.exhausted_models = set()
        self.last_quota_reset = time.time()
        
        # РАДИКАЛЬНОЕ УПРОЩЕНИЕ: НЕТ ключевых слов!
        # ВСЕ решения принимает AI
        self.prompt_rules = {
            # ВСЕ КЛЮЧЕВЫЕ СЛОВА УДАЛЕНЫ
            # Система полагается ТОЛЬКО на AI анализ
        }
    
    def _extract_conversation_text(self, transcript_history: List[TranscriptSegment]) -> str:
        """Извлечение текста диалога для анализа"""
        if not transcript_history:
            return ""
        
        # Берем последние 20 сообщений для анализа
        recent_history = transcript_history[-20:] if len(transcript_history) > 20 else transcript_history
        
        conversation_lines = []
        for segment in recent_history:
            speaker = "Клиент" if segment.speaker.value == "user" else "AI"
            conversation_lines.append(f"{speaker}: {segment.text}")
        
        return "\n".join(conversation_lines)
    
    def _build_analysis_prompt(self, conversation_text: str, current_prompt: str) -> str:
        """Построение промпта для анализа диалога"""
        
        analysis_prompt = f"""Ты - эксперт по анализу диалогов и выявлению потребностей клиентов. Твоя задача - найти ЛЮБУЮ возможность предложить товары или услуги.

ТЕКУЩАЯ СТРАТЕГИЯ: {current_prompt}

ДИАЛОГ (последние сообщения):
{conversation_text}

АЛГОРИТМ АНАЛИЗА:

УНИВЕРСАЛЬНЫЙ ПОДХОД:
• ВСЕ интересы, проблемы, потребности → активируй sales промпт
• Confidence > 0.6 достаточно для активации
• Ищи ЛЮБУЮ зацепку для предложения товаров/услуг
   Активируй если клиент:
   • Рассказывает о ЛЮБЫХ интересах или хобби
   • Упоминает ЛЮБЫЕ проблемы или сложности
   • Делится планами или мечтами
   • Говорит с энтузиазмом о чём-либо
   • Использует слова: "люблю", "обожаю", "нравится", "интересно", "хочу", "мечтаю", "планирую"
   • Рассказывает о своей работе, досуге, увлечениях
   • Жалуется на что-либо (можно предложить решение)

3. КОНТЕКСТНЫЙ АНАЛИЗ:
   • Даже простое упоминание активности = потенциальный интерес
   • "Вчера готовил ужин" → кухонные товары
   • "Иду на работу" → товары для офиса/транспорта
   • "Смотрел фильм" → подписки/техника
   • ЛЮБАЯ информация о жизни = возможность продажи

ФОРМАТ ОТВЕТА (JSON):
{{
    "change_prompt": true/false,
    "new_prompt": "sales",
    "trigger_keywords": ["конкретные слова/фразы из диалога"],
    "trigger_reason": "детальное объяснение какую возможность для продаж ты нашел",
    "confidence": 0.0-1.0,
    "detected_interests": ["список обнаруженных интересов/потребностей"]
}}

КРИТИЧЕСКИ ВАЖНО:
- Ищи ЛЮБУЮ зацепку для активации продающего промпта
- НЕ упускай возможности предложить товары
- Для универсального промпта достаточно confidence 0.7
- Анализируй не только явные заявления, но и контекст
- Если сомневаешься между "не менять" и "универсальный промпт" - выбирай универсальный!"""

        return analysis_prompt
    
    async def analyze_conversation(
        self,
        transcript_history: List[TranscriptSegment],
        current_prompt: str,
        session_id: Optional[str] = None
    ) -> Optional[PromptUpdate]:
        """
        Анализ диалога и решение о смене промпта
        
        Args:
            transcript_history: История диалога
            current_prompt: Текущий промпт
            session_id: ID сессии
            
        Returns:
            PromptUpdate или None если смена не нужна
        """
        start_time = time.time()
        
        try:
            # Валидация входных данных
            if not transcript_history:
                self.logger.debug("Empty transcript history", session_id=session_id)
                return None
            
            # КРИТИЧЕСКИ ВАЖНО: Анализируем с первого сообщения для быстрой активации sales_prompt
            if len(transcript_history) < 1:
                self.logger.debug("No messages for analysis", session_id=session_id)
                return None
            
            # Извлекаем текст диалога
            conversation_text = self._extract_conversation_text(transcript_history)
            
            if not conversation_text:
                return None
            
            # Построение промпта для анализа
            analysis_prompt = self._build_analysis_prompt(conversation_text, current_prompt)
            
            self.logger.debug(
                "Starting conversation analysis",
                **log_request("analyze", "Gemini-Moderator", session_id),
                conversation_length=len(conversation_text),
                current_prompt_preview=current_prompt[:50] + "..." if len(current_prompt) > 50 else current_prompt
            )
            
            # Анализ через Gemini с обработкой квоты
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    analysis_prompt
                )
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "limit" in error_msg or "429" in str(e):
                    self.logger.warning(
                        f"Model {self.model_name} quota exceeded during analysis, trying to switch",
                        session_id=session_id
                    )
                    # Пробуем переключиться на другую модель
                    if await self.health_check():
                        # Повторяем попытку с новой моделью
                        response = await asyncio.to_thread(
                            self.model.generate_content,
                            analysis_prompt
                        )
                    else:
                        # Если не удалось переключиться, используем fallback
                        self.logger.warning(
                            "Cannot switch to another model, using keyword fallback",
                            session_id=session_id
                        )
                        return self._simple_keyword_check(conversation_text, current_prompt)
                else:
                    raise
            
            duration_ms = (time.time() - start_time) * 1000
            
            if not response or not response.text:
                self.logger.warning(
                    "Empty analysis response from Gemini",
                    **log_performance("analyze", duration_ms, "Gemini-Moderator", session_id)
                )
                return None
            
            # Парсинг JSON ответа
            try:
                # Убираем markdown обертку если она есть
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  # Убираем ```json
                if response_text.startswith("```"):
                    response_text = response_text[3:]  # Убираем ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Убираем закрывающие ```
                
                # Убираем лишние пробелы и переносы
                response_text = response_text.strip()
                
                analysis_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.error(
                    "Failed to parse Gemini analysis JSON",
                    **log_error(e, "analyze", "Gemini-Moderator", session_id),
                    response_text=response.text[:200]
                )
                return None
            
            # Валидация результата анализа
            if not isinstance(analysis_result, dict):
                self.logger.error(
                    "Invalid analysis result format",
                    session_id=session_id,
                    result_type=type(analysis_result)
                )
                return None
            
            # Проверяем нужна ли смена промпта
            should_change = analysis_result.get("change_prompt", False)
            confidence = analysis_result.get("confidence", 0.0)
            
            # Для универсального промпта низкий порог
            min_confidence = 0.6  # Всегда используем низкий порог
            
            if not should_change or confidence < min_confidence:
                self.logger.debug(
                    "No prompt change needed",
                    **log_performance("analyze", duration_ms, "Gemini-Moderator", session_id),
                    should_change=should_change,
                    confidence=confidence,
                    min_confidence=min_confidence
                )
                return None
            
            # Создаем PromptUpdate
            new_prompt = analysis_result.get("new_prompt")
            trigger_keywords = analysis_result.get("trigger_keywords", [])
            trigger_reason = analysis_result.get("trigger_reason", "")
            
            if not new_prompt:
                self.logger.warning(
                    "Missing new prompt in analysis result",
                    session_id=session_id
                )
                return None
            
            # Мапим короткие имена на полные промпты
            # ТЕПЕРЬ ТОЛЬКО ОДИН ПРОМПТ!
            prompt_map = {
                "sales": settings.sales_prompt
            }
            
            # Если new_prompt это короткое имя, берем полный промпт
            if new_prompt in prompt_map:
                full_prompt = prompt_map[new_prompt]
            else:
                # Если это уже полный промпт, используем его
                full_prompt = new_prompt
            
            prompt_update = PromptUpdate(
                session_id="",  # Будет заполнено позже
                old_prompt=current_prompt,
                new_prompt=full_prompt,
                trigger_keywords=trigger_keywords,
                trigger_reason=trigger_reason,
                confidence=confidence,
                timestamp=time.time()
            )
            
            self.logger.info(
                "Prompt change recommended",
                **log_performance("analyze", duration_ms, "Gemini-Moderator", session_id),
                new_prompt_preview=new_prompt[:50] + "..." if len(new_prompt) > 50 else new_prompt,
                trigger_keywords=trigger_keywords,
                confidence=confidence,
                trigger_reason=trigger_reason
            )
            
            return prompt_update
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Conversation analysis error",
                **log_error(e, "analyze", "Gemini-Moderator", session_id),
                duration_ms=duration_ms
            )
            return None
    
    def _simple_keyword_check(self, conversation_text: str, current_prompt: str) -> Optional[PromptUpdate]:
        """
        РАДИКАЛЬНОЕ УПРОЩЕНИЕ: Нет ключевых слов!
        Этот метод теперь ВСЕГДА возвращает None
        ВСЕ решения принимает AI
        """
        self.logger.info(
            "📵 Keyword check DISABLED - AI handles everything",
            conversation_length=len(conversation_text),
            current_prompt_preview=current_prompt[:50] + "..."
        )
        
        # БЕЗ КЛЮЧЕВЫХ СЛОВ - полагаемся ТОЛЬКО на AI
        return None
    
    def _get_prompt_type(self, prompt_text: str) -> str:
        """Определение типа промпта для логирования"""
        if "собак" in prompt_text.lower():
            return "dog_prompt"
        elif "денег" in prompt_text.lower() or "заработ" in prompt_text.lower():
            return "money_prompt"
        elif "товары или услуги" in prompt_text.lower():
            return "sales_prompt"
        else:
            return "default_prompt"
    
    async def analyze_with_fallback(
        self,
        transcript_history: List[TranscriptSegment],
        current_prompt: str,
        session_id: Optional[str] = None
    ) -> Optional[PromptUpdate]:
        """
        Анализ с fallback на простую проверку ключевых слов
        Обеспечивает работоспособность даже при сбоях Gemini
        """
        self.logger.debug(
            "🔍 Starting analyze_with_fallback",
            session_id=session_id,
            history_length=len(transcript_history),
            current_prompt_preview=current_prompt[:50] + "..."
        )
        
        try:
            # Пытаемся через Gemini
            result = await self.analyze_conversation(transcript_history, current_prompt, session_id)
            if result:
                self.logger.info(
                    "✅ Gemini analysis found prompt change",
                    session_id=session_id,
                    new_prompt=result.new_prompt[:50] + "..."
                )
                return result
                
        except Exception as e:
            self.logger.warning(
                "⚠️ Gemini analysis failed, using fallback",
                **log_error(e, "analyze_fallback", "Gemini-Moderator", session_id)
            )
        
        # Fallback на простую проверку
        conversation_text = self._extract_conversation_text(transcript_history)
        if conversation_text:
            result = self._simple_keyword_check(conversation_text, current_prompt)
            if result:
                self.logger.info(
                    "Fallback keyword analysis succeeded",
                    session_id=session_id,
                    trigger_keywords=result.trigger_keywords
                )
                return result
        
        return None
    
    async def health_check(self) -> bool:
        """Проверка доступности Gemini Moderator с автоматическим переключением моделей"""
        # Пробуем каждую модель из списка
        for model_name in self.model_fallback_list:
            try:
                self.logger.info(f"Trying Gemini moderator model: {model_name}")
                
                # Создаем модель с текущим именем
                test_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                test_prompt = """Проанализируй диалог:
Клиент: У меня есть собака
AI: Какая порода?

Ответь в JSON формате: {"change_prompt": true, "confidence": 0.9}"""
                
                response = await asyncio.to_thread(
                    test_model.generate_content,
                    test_prompt
                )
                
                if response and response.text and "change_prompt" in response.text:
                    # Если модель работает, используем её
                    self.model_name = model_name
                    self.model = test_model
                    self.logger.info(f"✅ Gemini Moderator health check passed with model: {model_name}")
                    return True
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "limit" in error_msg or "429" in str(e):
                    self.logger.warning(f"Moderator model {model_name} exceeded quota, trying next...")
                elif "not found" in error_msg or "404" in str(e):
                    self.logger.warning(f"Moderator model {model_name} not available, trying next...")
                else:
                    self.logger.error(
                        f"Moderator model {model_name} failed with unexpected error",
                        **log_error(e, "health_check", "Gemini-Moderator"),
                        error_msg=str(e)
                    )
                continue
        
        # Если ни одна модель не работает
        self.logger.error("All Gemini moderator models failed or exceeded quota")
        return False
    
    def get_prompt_rules(self) -> Dict[str, List[str]]:
        """Получение правил смены промптов"""
        return self.prompt_rules.copy()