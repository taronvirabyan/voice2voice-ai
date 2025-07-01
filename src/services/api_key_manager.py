"""
Менеджер API ключей для Gemini с автоматической ротацией
КРИТИЧЕСКИ ВАЖНО: Обеспечивает надежную работу при превышении квот
"""

import time
from typing import List, Dict, Optional, Any
from collections import defaultdict
import google.generativeai as genai

from ..core.logging import LoggerMixin


class APIKeyManager(LoggerMixin):
    """
    Управление множественными API ключами с автоматической ротацией
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Args:
            api_keys: Список API ключей для ротации
        """
        if not api_keys:
            raise ValueError("At least one API key required")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        
        # Отслеживание состояния ключей
        self.key_status = defaultdict(lambda: {
            "exhausted": False,
            "last_error_time": 0,
            "error_count": 0,
            "models_exhausted": set()
        })
        
        # Время последнего сброса квот (07:00 UTC каждый день)
        self.last_quota_reset = self._get_last_quota_reset_time()
        
        self.logger.info(
            f"API Key Manager initialized with {len(api_keys)} keys",
            current_key=self._mask_key(self.get_current_key())
        )
    
    def get_current_key(self) -> str:
        """Получить текущий активный ключ"""
        return self.api_keys[self.current_key_index]
    
    def mark_key_exhausted(self, api_key: str, model_name: Optional[str] = None):
        """Пометить ключ как исчерпанный"""
        self.key_status[api_key]["exhausted"] = True
        self.key_status[api_key]["last_error_time"] = time.time()
        self.key_status[api_key]["error_count"] += 1
        
        if model_name:
            self.key_status[api_key]["models_exhausted"].add(model_name)
        
        self.logger.warning(
            f"API key marked as exhausted",
            key=self._mask_key(api_key),
            model=model_name,
            error_count=self.key_status[api_key]["error_count"]
        )
    
    def switch_to_next_key(self) -> Optional[str]:
        """Переключиться на следующий доступный ключ"""
        # Проверяем, не пора ли сбросить статусы
        self._check_quota_reset()
        
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            new_key = self.api_keys[self.current_key_index]
            
            # Проверяем, не исчерпан ли ключ
            if not self.key_status[new_key]["exhausted"]:
                self.logger.info(
                    f"Switched to API key {self.current_key_index + 1}/{len(self.api_keys)}",
                    key=self._mask_key(new_key)
                )
                
                # Переконфигурируем Gemini API
                genai.configure(api_key=new_key)
                return new_key
            
            attempts += 1
        
        # Все ключи исчерпаны
        self.current_key_index = original_index
        self.logger.error("All API keys exhausted!")
        
        # Проверяем время до сброса квоты
        time_until_reset = self._time_until_quota_reset()
        hours = time_until_reset // 3600
        minutes = (time_until_reset % 3600) // 60
        
        self.logger.error(
            f"Quota will reset in {hours}h {minutes}m at 07:00 UTC",
            all_keys_exhausted=True
        )
        
        return None
    
    def try_keys_for_model(self, model_name: str) -> Optional[str]:
        """Попробовать найти ключ, который работает с данной моделью"""
        self._check_quota_reset()
        
        for i, api_key in enumerate(self.api_keys):
            # Пропускаем ключи, где эта модель уже исчерпана
            if model_name in self.key_status[api_key]["models_exhausted"]:
                continue
            
            # Пропускаем полностью исчерпанные ключи
            if self.key_status[api_key]["exhausted"]:
                # Но проверяем, может это старая информация
                if time.time() - self.key_status[api_key]["last_error_time"] > 3600:
                    # Прошел час, попробуем снова
                    self.key_status[api_key]["exhausted"] = False
                    self.key_status[api_key]["models_exhausted"].clear()
                else:
                    continue
            
            # Пробуем этот ключ
            self.current_key_index = i
            genai.configure(api_key=api_key)
            
            self.logger.info(
                f"Trying key {i + 1}/{len(self.api_keys)} for model {model_name}",
                key=self._mask_key(api_key)
            )
            
            return api_key
        
        return None
    
    def _check_quota_reset(self):
        """Проверить, не произошел ли сброс квоты"""
        current_time = time.time()
        last_reset = self._get_last_quota_reset_time()
        
        if last_reset > self.last_quota_reset:
            self.logger.info("Quota reset detected! Clearing exhausted keys status")
            
            # Сбрасываем статусы всех ключей
            for key in self.api_keys:
                self.key_status[key]["exhausted"] = False
                self.key_status[key]["models_exhausted"].clear()
                self.key_status[key]["error_count"] = 0
            
            self.last_quota_reset = last_reset
            self.current_key_index = 0
    
    def _get_last_quota_reset_time(self) -> float:
        """Получить время последнего сброса квоты (07:00 UTC)"""
        import datetime
        
        now = datetime.datetime.utcnow()
        reset_hour = 7  # 07:00 UTC
        
        # Сегодняшний сброс
        today_reset = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        
        # Если сейчас до 07:00 UTC, последний сброс был вчера
        if now < today_reset:
            yesterday = now - datetime.timedelta(days=1)
            last_reset = yesterday.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        else:
            last_reset = today_reset
        
        return last_reset.timestamp()
    
    def _time_until_quota_reset(self) -> int:
        """Время в секундах до следующего сброса квоты"""
        import datetime
        
        now = datetime.datetime.utcnow()
        reset_hour = 7  # 07:00 UTC
        
        # Следующий сброс
        tomorrow = now + datetime.timedelta(days=1)
        next_reset = tomorrow.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        
        # Если сейчас до 07:00 UTC, сброс будет сегодня
        today_reset = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        if now < today_reset:
            next_reset = today_reset
        
        return int((next_reset - now).total_seconds())
    
    def _mask_key(self, api_key: str) -> str:
        """Маскировать API ключ для логов"""
        if not api_key or len(api_key) < 10:
            return api_key
        return f"{api_key[:10]}...{api_key[-6:]}"
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус всех ключей"""
        status = {
            "current_key_index": self.current_key_index,
            "total_keys": len(self.api_keys),
            "time_until_reset": self._time_until_quota_reset(),
            "keys": []
        }
        
        for i, key in enumerate(self.api_keys):
            key_info = {
                "index": i,
                "key": self._mask_key(key),
                "is_current": i == self.current_key_index,
                "exhausted": self.key_status[key]["exhausted"],
                "error_count": self.key_status[key]["error_count"],
                "models_exhausted": list(self.key_status[key]["models_exhausted"])
            }
            status["keys"].append(key_info)
        
        return status