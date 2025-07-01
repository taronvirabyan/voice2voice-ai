"""
Mock AI Service для демонстрации полной функциональности
КРИТИЧЕСКИ ВАЖНО: Обеспечивает 100% uptime для AI компонента
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional, List

from ..core.config import settings
from ..core.exceptions import AIServiceError
from ..core.logging import LoggerMixin, log_performance


class MockAIService(LoggerMixin):
    """
    Mock AI Service для демонстрации функциональности
    Генерирует интеллектуальные ответы без внешних API
    """
    
    def __init__(self):
        """Инициализация Mock AI сервиса"""
        self.model_name = "MockAI-v1.0"
        self.max_tokens = 150
        self.temperature = 0.7
        
        # Предустановленные ответы для разных типов запросов
        self.response_templates = {
            "greeting": [
                "Привет! Как дела? Чем могу помочь?",
                "Здравствуйте! Рад вас видеть!",
                "Добро пожаловать! Что вас интересует?",
            ],
            "question": [
                "Это интересный вопрос! Давайте разберемся.",
                "Хороший вопрос! Я думаю, что...",
                "Позвольте подумать над этим.",
            ],
            "default": [
                "Понятно! Это очень интересно.",
                "Спасибо за ваше сообщение!",
                "Я понял вас. Продолжайте, пожалуйста.",
                "Расскажите больше об этом.",
                "Интересная точка зрения!",
            ]
        }
        
        self.logger.info(
            "MockAI service initializing",
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
    
    async def health_check(self) -> bool:
        """Проверка здоровья Mock AI сервиса (всегда успешна)"""
        try:
            # Симулируем AI обработку
            test_response = await self._generate_response("Тест", "test_session")
            
            success = test_response is not None and len(test_response.strip()) > 0
            
            self.logger.info(
                "MockAI health check completed",
                success=success,
                test_response_length=len(test_response) if test_response else 0
            )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "MockAI health check failed",
                error=str(e)
            )
            return False
    
    async def generate_response(
        self,
        user_message: str = None,
        current_prompt: str = None,
        conversation_history: List = None,
        session_id: str = None,
        # Для обратной совместимости
        text: str = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Генерация ответа на текст пользователя
        
        Args:
            text: Входной текст от пользователя
            session_id: ID сессии
            context: Дополнительный контекст
            
        Returns:
            str: Сгенерированный ответ
        """
        # Поддержка обоих форматов вызова
        if user_message is None and text is not None:
            user_message = text
        if user_message is None:
            raise ValueError("No user message provided")
            
        self.logger.info(
            "MockAI generating response",
            session_id=session_id,
            text_length=len(user_message),
            text_preview=user_message[:50] + "..." if len(user_message) > 50 else user_message,
            has_prompt=current_prompt is not None,
            prompt_preview=current_prompt[:50] + "..." if current_prompt and len(current_prompt) > 50 else current_prompt
        )
        
        try:
            response = await self._generate_response(user_message, session_id, context, current_prompt)
            
            self.logger.info(
                "✅ MockAI response generated",
                session_id=session_id,
                response_length=len(response),
                response_preview=response[:50] + "..." if len(response) > 50 else response
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "MockAI response generation failed",
                error=str(e),
                session_id=session_id,
                text_length=len(text)
            )
            raise AIServiceError(f"Mock AI response generation failed: {str(e)}")
    
    async def _generate_response(
        self,
        text: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        current_prompt: str = None
    ) -> str:
        """
        Внутренняя генерация ответа с интеллектуальной логикой
        """
        start_time = time.time()
        
        # Симулируем время обработки AI
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Анализируем тип входного сообщения
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["привет", "здравствуй", "добро", "hello"]):
            category = "greeting"
        elif "?" in text or any(word in text_lower for word in ["что", "как", "почему", "когда", "где"]):
            category = "question"
        else:
            category = "default"
        
        # Выбираем базовый ответ
        base_response = random.choice(self.response_templates[category])
        
        # Используем промпт для генерации соответствующего ответа
        if current_prompt:
            prompt_lower = current_prompt.lower()
            
            # Промпт про собак
            if "собак" in prompt_lower and "gps" in prompt_lower:
                if "собак" in text_lower or "пес" in text_lower or "щен" in text_lower:
                    base_response = "Отлично, что у вас есть собака! Какая порода? Кстати, видел классные GPS-ошейники для собак - всегда знаешь где питомец."
                else:
                    base_response = "Круто! А у вас есть домашние животные? Я недавно узнал про умные устройства для питомцев."
            
            # Промпт про деньги
            elif "денег" in prompt_lower and "заработ" in prompt_lower:
                if "денег" in text_lower or "финанс" in text_lower:
                    base_response = "Понимаю, с финансами бывает сложно. Знаете, многие сейчас зарабатывают на фрилансе или продают вещи онлайн."
                else:
                    base_response = "Интересно! Кстати, сейчас много возможностей для дополнительного заработка. Хотите узнать?"
            
            # Промпт про детей
            elif "дет" in prompt_lower and "развив" in prompt_lower:
                if "дет" in text_lower or "ребен" in text_lower:
                    base_response = "Здорово, что у вас есть дети! Сейчас столько развивающих игр и гаджетов для детей - от интерактивных глобусов до умных часов."
                else:
                    base_response = "Понятно! А есть ли у вас дети? Просто сейчас столько интересных развивающих штук появилось."
            
            # Универсальный продающий промпт
            elif "товар" in prompt_lower or "услуг" in prompt_lower:
                base_response = self._generate_sales_response(text_lower)
        else:
            # Старая логика для обратной совместимости
            if "собак" in text_lower:
                base_response += " Кстати, если у вас есть собака, вам могут понадобиться аксессуары для неё."
            elif "деньги" in text_lower or "финанс" in text_lower:
                base_response += " Если вас интересуют финансовые возможности, у меня есть предложения."
            elif "работа" in text_lower:
                base_response += " Расскажите больше о ваших профессиональных интересах."
        
        # Добавляем вариативность в ответы
        if random.random() < 0.3:  # 30% шанс добавить дополнительную фразу
            additional_phrases = [
                "Что вы об этом думаете?",
                "Интересно узнать ваше мнение.",
                "Хотели бы узнать подробнее?",
                "Есть ли у вас вопросы?",
            ]
            base_response += " " + random.choice(additional_phrases)
        
        generation_time = time.time() - start_time
        
        self.logger.info(
            "Response generation completed",
            **log_performance("mock_ai_generation", generation_time * 1000, "MockAI"),
            text_length=len(text),
            response_length=len(base_response),
            category=category,
            session_id=session_id
        )
        
        return base_response
    
    def _generate_sales_response(self, text_lower: str) -> str:
        """Генерация продающего ответа на основе темы"""
        if "спорт" in text_lower:
            return "Здорово, что вы занимаетесь спортом! Кстати, видел классные фитнес-трекеры и умные гантели. Интересно?"
        elif "готов" in text_lower or "кухн" in text_lower:
            return "О, вы любите готовить! Знаете про новые кухонные гаджеты? Мультиварки с приложениями, умные весы..."
        elif "путешеств" in text_lower:
            return "Путешествия - это прекрасно! Кстати, есть отличные компактные гаджеты для путешествий. Хотите расскажу?"
        elif "работ" in text_lower:
            return "Работа важна! Знаете, есть классные штуки для продуктивности - умные планировщики, эргономичные аксессуары."
        else:
            return "Интересно! Расскажите больше о ваших увлечениях - наверняка смогу посоветовать что-то полезное."
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Информация о Mock AI сервисе"""
        return {
            "service": "MockAI",
            "model": self.model_name,
            "capabilities": [
                "Text generation",
                "Context awareness",
                "Russian language support",
                "Real-time responses",
                "Category-based responses"
            ],
            "features": [
                "100% uptime",
                "No external API dependencies",
                "Personalized responses",
                "Session context",
                "Performance metrics"
            ],
            "response_categories": list(self.response_templates.keys()),
            "average_response_time_ms": "100-500"
        }
    
    async def shutdown(self):
        """Корректное завершение работы сервиса"""
        self.logger.info("MockAI service shutting down")