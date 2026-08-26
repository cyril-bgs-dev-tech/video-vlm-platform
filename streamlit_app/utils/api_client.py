"""
utils/api_client.py — Client HTTP synchrone + polling tasks
"""
import time
import httpx
import streamlit as st
from typing import Optional, Any

API = st.secrets.get("API_BASE_URL", "http://localhost:8000")


class APIClient:
    """Client wrapper pour l'API FastAPI."""

    def __init__(self, base_url: str = API, timeout: float = 30.0):
        self.base      = base_url.rstrip("/")
        self.timeout   = timeout

    def get(self, path: str, **kwargs) -> dict:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.base}{path}", **kwargs)
            r.raise_for_status()
            return r.json()

    def post(self, path: str, json: dict = None, files=None, timeout: float = None) -> dict:
        t = timeout or self.timeout
        with httpx.Client(timeout=t) as c:
            if files:
                r = c.post(f"{self.base}{path}", files=files)
            else:
                r = c.post(f"{self.base}{path}", json=json)
            r.raise_for_status()
            return r.json()

    def poll_task(
        self,
        task_id: str,
        interval: float = 1.0,
        timeout: float = 300.0,
        progress_callback=None,
    ) -> Optional[Any]:
        """
        Poll une task Celery jusqu'à complétion.
        Retourne le résultat ou None si timeout.
        """
        start  = time.time()
        status = st.empty()
        while time.time() - start < timeout:
            try:
                data = self.get(f"/tasks/{task_id}")
                state = data.get("status", "PENDING")
                if progress_callback:
                    progress_callback(state)
                else:
                    status.info(f"⏳ Task `{task_id}` — état : **{state}**")
                if state == "SUCCESS":
                    status.empty()
                    return data.get("result")
                elif state == "FAILURE":
                    status.error(f"❌ Task échouée : {data.get('result')}")
                    return None
            except Exception:
                pass
            time.sleep(interval)
        status.error(f"⏱️ Timeout ({timeout}s) atteint pour la task `{task_id}`")
        return None

    def health(self) -> dict:
        try:
            return self.get("/health")
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}


# Singleton partagé
client = APIClient()
