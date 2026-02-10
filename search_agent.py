#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO SEARCH AGENT - Complete với lazy loading và caching
"""
import time
import json
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lazy_loader import get_lazy_loader


class DracoSearchAgent:
    """Search agent với lazy loading và intelligent caching"""

    def __init__(self, config):
        self.config = config
        self.initialized = False

        # Lazy loader
        self.lazy_loader = get_lazy_loader()

        # Cache system
        self.cache = {}
        self.cache_file = None
        self.cache_lock = threading.RLock()

        # Performance
        self.last_search_time = 0
        self.min_search_interval = 2.0

        # Initialize
        self.initialize()

    def initialize(self):
        """Khởi tạo search agent với lazy loading"""
        if self.initialized:
            return True

        try:
            print("🔍 Initializing Search Agent (Lazy Loading)...")

            # Register search engine component
            self._register_components()

            # Load cache
            self._load_cache()

            self.initialized = True
            print("✅ Search Agent initialized")
            return True

        except Exception as e:
            print(f"❌ Search Agent initialization failed: {e}")
            return False

    def _register_components(self):
        """Đăng ký components cho lazy loading"""

        def load_search_engine():
            """Load search engine on-demand"""
            try:
                from duckduckgo_search import DDGS
                engine = DDGS()
                print("✅ Search Engine loaded (DuckDuckGo)")
                return engine
            except ImportError:
                print("❌ DuckDuckGo Search not installed")
                raise

        # Đăng ký search engine
        self.lazy_loader.register_component(
            name="search_engine",
            loader_func=load_search_engine,
            estimated_memory_mb=10
        )

    def _load_cache(self):
        """Load search cache từ file"""
        try:
            cache_dir = self.config.get_storage_paths()["cache"]
            self.cache_file = cache_dir / "search_cache.json"

            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)

                # Clean old cache entries
                self._clean_cache()
                print(f"✅ Loaded {len(self.cache)} cached search results")
            else:
                self.cache = {}

        except Exception as e:
            print(f"Failed to load cache: {e}")
            self.cache = {}

    def _clean_cache(self):
        """Dọn dẹp cache cũ"""
        cache_duration = self.config.get("search.cache_duration", 3600)
        cutoff_time = time.time() - cache_duration

        with self.cache_lock:
            expired_keys = []

            for key, entry in self.cache.items():
                if entry.get("timestamp", 0) < cutoff_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

    def search(self, query: str, max_results: int = 5,
               use_cache: bool = True) -> Dict[str, Any]:
        """Tìm kiếm thông tin với lazy loading"""

        # Rate limiting
        current_time = time.time()
        if current_time - self.last_search_time < self.min_search_interval:
            time.sleep(self.min_search_interval - (current_time - self.last_search_time))

        # Check cache
        cache_key = f"{query}_{max_results}"

        if use_cache and cache_key in self.cache:
            cached_result = self.cache[cache_key]

            # Check if cache is still valid
            cache_age = time.time() - cached_result.get("timestamp", 0)
            cache_duration = self.config.get("search.cache_duration", 3600)

            if cache_age < cache_duration:
                return {
                    "success": True,
                    "query": query,
                    "results": cached_result["results"],
                    "from_cache": True,
                    "cache_age": cache_age
                }

        try:
            # Load search engine on-demand
            search_engine = self.lazy_loader.get_component("search_engine")

            print(f"🔍 Searching for: {query}")

            # Perform search
            results = []
            search_results = search_engine.text(query, max_results=max_results)

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "duckduckgo"
                })

            # Update cache
            with self.cache_lock:
                self.cache[cache_key] = {
                    "query": query,
                    "results": results,
                    "timestamp": time.time(),
                    "result_count": len(results)
                }

                # Save cache to file
                self._save_cache()

            self.last_search_time = time.time()

            # Schedule unload
            self.lazy_loader.schedule_unload("search_engine", timeout=30)

            return {
                "success": True,
                "query": query,
                "results": results,
                "from_cache": False,
                "result_count": len(results)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def search_news(self, topic: str = "technology", max_results: int = 5) -> Dict[str, Any]:
        """Tìm kiếm tin tức"""
        import datetime
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        query = f"{topic} news {today}"
        return self.search(query, max_results)

    def search_weather(self, location: str = None) -> Dict[str, Any]:
        """Tìm kiếm thời tiết"""
        if not location:
            # Try to detect location from IP
            try:
                import requests
                response = requests.get('https://ipinfo.io', timeout=5)
                location_data = response.json()
                location = location_data.get('city', 'Ho Chi Minh City')
            except:
                location = "Ho Chi Minh City"

        query = f"weather {location} today"
        result = self.search(query, max_results=1)

        if result.get("success") and result.get("results"):
            # Parse weather info từ kết quả
            snippet = result["results"][0]["snippet"]

            # Simple parsing
            if "°" in snippet or "C" in snippet or "F" in snippet:
                return {
                    "success": True,
                    "location": location,
                    "weather_info": snippet,
                    "source": result["results"][0]["url"]
                }

        # Fallback search
        return self.search(f"{location} weather forecast", max_results=2)

    def search_wikipedia(self, topic: str) -> Dict[str, Any]:
        """Tìm kiếm Wikipedia"""
        query = f"{topic} wikipedia"
        return self.search(query, max_results=3)

    def search_youtube(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Tìm kiếm YouTube videos"""
        search_query = f"{query} site:youtube.com"
        return self.search(search_query, max_results)

    def search_images(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Tìm kiếm hình ảnh"""
        try:
            # Load search engine
            search_engine = self.lazy_loader.get_component("search_engine")

            results = []
            image_results = search_engine.images(query, max_results=max_results)

            for img in image_results:
                results.append({
                    "title": img.get("title", ""),
                    "url": img.get("image", ""),
                    "source": img.get("source", ""),
                    "thumbnail": img.get("thumbnail", "")
                })

            # Schedule unload
            self.lazy_loader.schedule_unload("search_engine", timeout=30)

            return {
                "success": True,
                "query": query,
                "results": results,
                "result_count": len(results)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _save_cache(self):
        """Lưu cache vào file"""
        try:
            with self.cache_lock:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        with self.cache_lock:
            total_entries = len(self.cache)

            # Calculate cache age distribution
            now = time.time()
            age_distribution = {
                "last_hour": 0,
                "last_day": 0,
                "older": 0
            }

            for entry in self.cache.values():
                age = now - entry.get("timestamp", 0)

                if age < 3600:
                    age_distribution["last_hour"] += 1
                elif age < 86400:
                    age_distribution["last_day"] += 1
                else:
                    age_distribution["older"] += 1

            return {
                "total_entries": total_entries,
                "age_distribution": age_distribution,
                "cache_file": str(self.cache_file) if self.cache_file else None
            }

    def clear_cache(self):
        """Xóa cache"""
        with self.cache_lock:
            self.cache.clear()

            if self.cache_file and self.cache_file.exists():
                self.cache_file.unlink()

            print("Search cache cleared")

    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái search agent"""
        loader_status = self.lazy_loader.get_status()
        cache_stats = self.get_cache_stats()

        return {
            "initialized": self.initialized,
            "loader_status": loader_status.get("search_engine", {}),
            "cache_stats": cache_stats
        }

    def cleanup(self):
        """Dọn dẹp search agent"""
        self.lazy_loader.unload_all()
        self._save_cache()
        print("Search Agent cleaned up")