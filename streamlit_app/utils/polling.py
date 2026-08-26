"""
streamlit_app/utils/polling.py — Polling de tasks Celery depuis Streamlit
Supporte SSE (Server-Sent Events) et polling classique httpx.
Usage :
    from utils.polling import poll_task, poll_task_sse

    # Option 1 — Polling simple (bloquant pendant max timeout secondes)
    result = poll_task(task_id, api_base, timeout=120)

    # Option 2 — Polling avec barre de progression Streamlit
    result = poll_task_with_ui(task_id, api_base, label="SAM2 en cours…")
"""
import time
from typing import Optional, Any

import httpx
import streamlit as st


def poll_task(
    task_id:  str,
    api_base: str,
    timeout:  float = 300.0,
    interval: float = 1.0,
) -> Optional[Any]:
    """
    Interroge /tasks/{task_id} jusqu'à complétion.
    Retourne le résultat ou None en cas d'échec / timeout.
    Bloquant — à appeler dans un st.spinner() ou un thread.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{api_base}/tasks/{task_id}", timeout=10)
            if resp.status_code == 200:
                data   = resp.json()
                status = data.get("status", "PENDING")
                if status == "SUCCESS":
                    return data.get("result")
                elif status == "FAILURE":
                    return {"error": data.get("error", "Task échouée")}
                elif status == "REVOKED":
                    return {"error": "Task annulée"}
        except Exception:
            pass
        time.sleep(interval)
    return {"error": f"Timeout après {timeout}s"}


def poll_task_with_ui(
    task_id:    str,
    api_base:   str,
    label:      str  = "Traitement en cours…",
    timeout:    float = 300.0,
    interval:   float = 1.0,
) -> Optional[Any]:
    """
    Polling avec barre de progression Streamlit.
    À appeler dans le corps principal d'une page (pas dans un thread).
    """
    status_ph   = st.empty()
    progress_ph = st.progress(0)
    deadline    = time.time() + timeout
    step        = 0

    while time.time() < deadline:
        elapsed  = time.time() - (deadline - timeout)
        fraction = min(elapsed / timeout, 0.95)
        progress_ph.progress(fraction, f"{label} ({int(fraction*100)}%)")

        try:
            resp = httpx.get(f"{api_base}/tasks/{task_id}", timeout=10)
            if resp.status_code == 200:
                data   = resp.json()
                status = data.get("status", "PENDING")
                status_ph.caption(f"État : {status}")

                if "progress" in data and data["progress"] is not None:
                    progress_ph.progress(
                        data["progress"] / 100,
                        f"{label} ({data['progress']}%)"
                    )

                if status == "SUCCESS":
                    progress_ph.progress(1.0, "✅ Terminé")
                    status_ph.empty()
                    return data.get("result")
                elif status in ("FAILURE", "REVOKED"):
                    progress_ph.empty()
                    status_ph.error(f"❌ {data.get('error', status)}")
                    return {"error": data.get("error", status)}
        except httpx.ConnectError:
            status_ph.warning("API non joignable…")
        except Exception as e:
            status_ph.caption(f"Erreur polling : {e}")

        time.sleep(interval)
        step += 1

    progress_ph.empty()
    status_ph.error(f"⏱️ Timeout après {timeout}s")
    return {"error": "timeout"}


def launch_and_poll(
    api_base:    str,
    endpoint:    str,
    payload:     dict,
    label:       str   = "Traitement…",
    post_timeout: float = 30.0,
    poll_timeout: float = 300.0,
) -> Optional[Any]:
    """
    Lance une tâche via POST puis la poll jusqu'à complétion.
    Gère les erreurs de connexion API proprement.

    Retourne le résultat ou un dict {"error": "..."}.
    """
    # 1. Lancer la tâche
    try:
        resp = httpx.post(
            f"{api_base}{endpoint}",
            json=payload,
            timeout=post_timeout,
        )
    except httpx.ConnectError:
        return {"error": f"API non joignable : {api_base}"}
    except Exception as e:
        return {"error": str(e)}

    if resp.status_code not in (200, 202):
        return {"error": f"HTTP {resp.status_code} : {resp.text[:200]}"}

    task_id = resp.json().get("task_id")
    if not task_id:
        # Réponse synchrone (pas de task_id) — retourner directement
        return resp.json()

    # 2. Poller
    return poll_task_with_ui(
        task_id=task_id,
        api_base=api_base,
        label=label,
        timeout=poll_timeout,
    )


def store_result(key: str, result: Any) -> bool:
    """Stocke un résultat dans st.session_state pour l'afficher dans les pages."""
    if result and "error" not in result:
        # Normaliser : les workers retournent souvent {"results": [...], "video_id": "..."}
        if isinstance(result, dict) and "results" in result:
            st.session_state[key] = result["results"]
        else:
            st.session_state[key] = result
        return True
    return False
