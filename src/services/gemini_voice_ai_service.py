"""
Gemini Voice AI сервис для диалога с клиентом
КРИТИЧЕСКИ ВАЖНО: Оптимизирован для корректной работы с русским языком
"""

import asyncio
import time
from typing import List, Dict, Optional
import google.generativeai as genai

from ..core.config import settings
from ..core.models import TranscriptSegment, MessageRole
from ..core.exceptions import ValidationError
from ..core.logging import LoggerMixin, log_request, log_error, log_performance


class GeminiVoiceAIService(LoggerMixin):
    """
    Сервис диалога с клиентом через Gemini API
    Оптимизирован для естественного voice2voice общения
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
        
        # Используем первую работающую модель из списка
        self.model_name = self.model_fallback_list[0]  # gemini-2.0-flash-exp
        
        # Настройки генерации для диалога
        self.generation_config = genai.types.GenerationConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            top_p=0.8,
            top_k=40
        )
        
        # Настройки безопасности (разрешаем все для корректной работы диалога)
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
        self.quota_reset_time = None
        self.last_quota_reset = time.time()
    
    def _validate_input(self, user_message: str, current_prompt: str) -> None:
        """Валидация входных данных"""
        if not user_message or not user_message.strip():
            raise ValidationError("Empty user message", "user_message", user_message)
        
        if not current_prompt or not current_prompt.strip():
            raise ValidationError("Empty current prompt", "current_prompt", current_prompt)
        
        if len(user_message) > 1000:
            raise ValidationError("User message too long", "user_message", len(user_message))
    
    def _build_conversation_context(
        self,
        user_message: str,
        current_prompt: str,
        conversation_history: List[TranscriptSegment],
        max_history: int = 10
    ) -> str:
        """Построение контекста диалога для Gemini"""
        
        # Системный промпт
        system_context = f"""Ты - дружелюбный собеседник, который ведет голосовой диалог на русском языке.

ТЕКУЩАЯ ЗАДАЧА: {current_prompt}

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском языке
2. Используй разговорный стиль, как в живом диалоге
3. Ответ должен быть коротким (1-2 предложения максимум)
4. Следуй текущей задаче, но делай это естественно и ненавязчиво
5. Будь искренне заинтересован в собеседнике
6. Не упоминай что ты AI или что это чат
7. Отвечай так, как будто разговариваешь по телефону с хорошим знакомым

История последних сообщений:"""
        
        # Добавляем историю диалога
        recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
        
        for segment in recent_history:
            speaker = "Ты" if segment.speaker == MessageRole.ASSISTANT else "Собеседник"
            system_context += f"\n{speaker}: {segment.text}"
        
        # Добавляем текущее сообщение
        system_context += f"\nСобеседник: {user_message}"
        system_context += f"\nТы:"
        
        return system_context
    
    async def generate_response(
        self,
        user_message: str,
        current_prompt: str,
        conversation_history: List[TranscriptSegment],
        session_id: Optional[str] = None
    ) -> str:
        """
        Генерация ответа AI собеседника
        
        Args:
            user_message: Сообщение от пользователя
            current_prompt: Текущий промпт/задача
            conversation_history: История диалога
            session_id: ID сессии для логирования
            
        Returns:
            str: Ответ AI для синтеза речи
        """
        start_time = time.time()
        
        try:
            # Проверяем необходимость сброса кэша исчерпанных моделей (каждый час)
            if time.time() - self.last_quota_reset > 3600:  # 1 час
                self.logger.info("Resetting exhausted models cache (hourly reset)")
                self.exhausted_models.clear()
                self.last_quota_reset = time.time()
            
            # Валидация входных данных
            self._validate_input(user_message, current_prompt)
            
            # Построение контекста
            conversation_context = self._build_conversation_context(
                user_message, current_prompt, conversation_history
            )
            
            self.logger.debug(
                "Generating Gemini response",
                **log_request("generate", "Gemini", session_id),
                prompt_length=len(current_prompt),
                message_length=len(user_message),
                history_count=len(conversation_history)
            )
            
            # Генерация ответа через Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                conversation_context
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Проверка ответа
            if not response or not response.text:
                self.logger.warning(
                    "Empty response from Gemini",
                    **log_performance("generate", duration_ms, "Gemini", session_id)
                )
                return "Извините, не расслышал. Можете повторить?"
            
            generated_text = response.text.strip()
            
            # Постобработка ответа для улучшения качества диалога
            generated_text = self._postprocess_response(generated_text)
            
            self.logger.info(
                "Gemini response generated successfully",
                **log_performance("generate", duration_ms, "Gemini", session_id),
                response_length=len(generated_text),
                input_tokens=len(conversation_context.split()),
                output_tokens=len(generated_text.split())
            )
            
            return generated_text
            
        except ValidationError as e:
            self.logger.error(
                "Gemini input validation error",
                **log_error(e, "generate", "Gemini", session_id)
            )
            return "Извините, произошла ошибка. Попробуйте еще раз."
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e).lower()
            
            # Проверяем, если это ошибка квоты
            # ResourceExhausted - имя класса ошибки, не в тексте!
            if ("quota" in error_msg or 
                "limit" in error_msg or 
                "429" in str(e) or 
                type(e).__name__ == "ResourceExhausted" or
                "resourceexhausted" in error_msg):
                self.logger.warning(
                    f"Model {self.model_name} quota exceeded, attempting fallback",
                    **log_error(e, "generate", "Gemini", session_id),
                    current_model=self.model_name
                )
                
                # Пытаемся переключиться на следующую модель
                current_index = self.model_fallback_list.index(self.model_name) if self.model_name in self.model_fallback_list else -1
                
                # Добавляем текущую модель в исчерпанные
                self.exhausted_models.add(self.model_name)
                
                # Перебираем оставшиеся модели
                for i in range(current_index + 1, len(self.model_fallback_list)):
                    try:
                        next_model = self.model_fallback_list[i]
                        
                        # Пропускаем исчерпанные модели
                        if next_model in self.exhausted_models:
                            self.logger.debug(f"Skipping exhausted model: {next_model}")
                            continue
                            
                        self.logger.info(f"Trying fallback model: {next_model}")
                        
                        # Создаем новую модель
                        self.model = genai.GenerativeModel(
                            model_name=next_model,
                            generation_config=self.generation_config,
                            safety_settings=self.safety_settings
                        )
                        self.model_name = next_model
                        
                        # Пытаемся сгенерировать ответ с новой моделью
                        response = await asyncio.to_thread(
                            self.model.generate_content,
                            conversation_context
                        )
                        
                        if response and response.text:
                            generated_text = response.text.strip()
                            generated_text = self._postprocess_response(generated_text)
                            
                            self.logger.info(
                                f"✅ Successfully used fallback model: {next_model}",
                                response_length=len(generated_text)
                            )
                            
                            return generated_text
                            
                    except Exception as fallback_error:
                        error_str = str(fallback_error).lower()
                        if ("quota" in error_str or 
                            "429" in error_str or 
                            type(fallback_error).__name__ == "ResourceExhausted" or
                            "resourceexhausted" in error_str):
                            self.exhausted_models.add(next_model)
                            self.logger.warning(
                                f"Fallback model {next_model} also exceeded quota",
                                error=str(fallback_error)
                            )
                        else:
                            self.logger.warning(
                                f"Fallback model {next_model} failed with error",
                                error=str(fallback_error)
                            )
                        continue
                
                # Если все модели исчерпаны
                self.logger.error(
                    "All Gemini models exhausted",
                    tried_models=self.model_fallback_list[current_index:],
                    duration_ms=duration_ms
                )
                return "Извините, сервис временно перегружен. Попробуйте через несколько минут."
            
            # Для других ошибок - расширенное логирование
            self.logger.error(
                "Gemini generation error (NOT quota related)",
                error_class=type(e).__name__,  # Изменено с error_type на error_class
                error_str=str(e),
                error_repr=repr(e),
                model_name=self.model_name,
                **log_error(e, "generate", "Gemini", session_id),
                duration_ms=duration_ms
            )
            
            # Логируем полный traceback
            import traceback
            self.logger.error(
                "Full traceback:",
                traceback=traceback.format_exc()
            )
            
            return "Простите, временные технические неполадки. Продолжим диалог."
    
    def _postprocess_response(self, text: str) -> str:
        """Постобработка ответа для улучшения качества диалога"""
        
        # Удаляем лишние символы и форматирование
        text = text.replace("*", "").replace("**", "").replace("_", "")
        text = text.replace("\n", " ").replace("\r", " ")
        
        # Убираем множественные пробелы
        while "  " in text:
            text = text.replace("  ", " ")
        
        # Ограничиваем длину для голосового диалога
        sentences = text.split(". ")
        if len(sentences) > 2:
            text = ". ".join(sentences[:2]) + "."
        
        # Убираем префиксы если Gemini их добавил
        prefixes_to_remove = [
            "Ты: ", "Собеседник: ", "AI: ", "Ответ: ",
            "Скажу так: ", "Отвечу: "
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Проверяем минимальную длину
        if len(text.strip()) < 3:
            return "Понятно. Расскажите подробнее."
        
        return text.strip()
    
    async def generate_batch_responses(
        self,
        messages: List[str],
        current_prompt: str,
        session_id: str
    ) -> List[str]:
        """
        Пакетная генерация ответов для оптимизации
        """
        self.logger.info(
            "Starting batch response generation",
            session_id=session_id,
            batch_size=len(messages)
        )
        
        tasks = [
            self.generate_response(msg, current_prompt, [], session_id)
            for msg in messages
        ]
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем результаты
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    self.logger.error(
                        "Batch generation item failed",
                        session_id=session_id,
                        item_index=i,
                        error=str(response)
                    )
                    processed_responses.append("Извините, не понял. Повторите пожалуйста.")
                else:
                    processed_responses.append(response)
            
            self.logger.info(
                "Batch generation completed",
                session_id=session_id,
                successful_items=sum(1 for r in responses if not isinstance(r, Exception)),
                failed_items=sum(1 for r in responses if isinstance(r, Exception))
            )
            
            return processed_responses
            
        except Exception as e:
            self.logger.error(
                "Batch generation error",
                **log_error(e, "batch_generate", "Gemini", session_id)
            )
            return ["Извините, технические неполадки." for _ in messages]
    
    async def health_check(self) -> bool:
        """Проверка доступности Gemini API с автоматическим переключением моделей"""
        # Пробуем каждую модель из списка
        for model_name in self.model_fallback_list:
            try:
                self.logger.info(f"Trying Gemini model: {model_name}")
                
                # Создаем модель с текущим именем
                test_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                test_response = await asyncio.to_thread(
                    test_model.generate_content,
                    "Скажи 'тест' на русском языке"
                )
                
                if test_response and test_response.text and "тест" in test_response.text.lower():
                    # Если модель работает, используем её
                    self.model_name = model_name
                    self.model = test_model
                    self.logger.info(f"✅ Gemini health check passed with model: {model_name}")
                    return True
                    
            except Exception as e:
                error_msg = str(e).lower()
                if ("quota" in error_msg or 
                    "limit" in error_msg or 
                    "429" in str(e) or
                    type(e).__name__ == "ResourceExhausted"):
                    self.logger.warning(f"Model {model_name} exceeded quota, trying next...")
                elif "not found" in error_msg or "404" in str(e):
                    self.logger.warning(f"Model {model_name} not available, trying next...")
                else:
                    self.logger.error(
                        f"Model {model_name} failed with unexpected error",
                        **log_error(e, "health_check", "Gemini"),
                        error_msg=str(e)
                    )
                continue
        
        # Если ни одна модель не работает
        self.logger.error("All Gemini models failed or exceeded quota")
        return False
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Получение статистики использования"""
        return {
            "model": self.model_name,
            "temperature": self.generation_config.temperature,
            "max_tokens": self.generation_config.max_output_tokens,
            "safety_settings": len(self.safety_settings)
        }