"""
utils/ws_client.py — Client WebSocket pour Streamlit
FIXED: asyncio.run() remplacé par nouvelle event loop dans thread (compatible Streamlit)
"""
import json
import asyncio
import threading
import streamlit as st
from typing import Callable, Optional

WS_BASE = st.secrets.get("WS_BASE_URL", "ws://localhost:8000")


class WSClient:
    """
    Client WebSocket non-bloquant pour Streamlit.
    Crée une boucle asyncio DANS le thread dédié pour éviter le conflit
    avec la boucle asyncio déjà gérée par Streamlit dans le thread principal.
    """

    def __init__(self, base_url: str = WS_BASE):
        self.base    = base_url.rstrip("/")
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    def start_detection_stream(
        self,
        video_id:   str,
        model:      str,
        query:      str,
        fps:        int,
        on_message: Callable[[dict], None],
        on_done:    Optional[Callable[[], None]] = None,
        on_error:   Optional[Callable[[str], None]] = None,
    ):
        """Démarre un stream dans un thread background avec sa propre event loop."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_in_new_loop,
            args=(video_id, model, query, fps, on_message, on_done, on_error),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Signale l'arrêt au thread WebSocket."""
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_in_new_loop(self, video_id, model, query, fps, on_msg, on_done, on_err):
        """
        Crée une nouvelle boucle asyncio dans ce thread.
        N'utilise PAS asyncio.run() pour éviter le conflit avec Streamlit.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._stream(video_id, model, query, fps, on_msg, on_done, on_err)
            )
        except Exception as e:
            if on_err:
                on_err(str(e))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _stream(self, video_id, model, query, fps, on_msg, on_done, on_err):
        try:
            import websockets
            uri = f"{self.base}/ws/detect/{video_id}"
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
            ) as ws:
                await ws.send(json.dumps({"model": model, "query": query, "fps": fps}))
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    data = json.loads(raw)
                    on_msg(data)
                    if data.get("type") in ("done", "error"):
                        break
            if on_done:
                on_done()
        except Exception as e:
            if on_err:
                on_err(str(e))


def render_ws_status(
    video_id: str,
    model:    str = "grounding_dino",
    query:    str = "person",
    fps:      int = 2,
) -> list:
    """
    Composant Streamlit réutilisable pour afficher un stream de détection.

    Usage:
        from utils.ws_client import render_ws_status
        results = render_ws_status(video_id, model="yolo_world_v2_x", query="car", fps=3)
    """
    # Initialisation du session state
    defaults = {
        "ws_client":      None,
        "ws_running":     False,
        "ws_detections":  [],
        "ws_frame_count": 0,
        "ws_error":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state["ws_client"] is None:
        st.session_state["ws_client"] = WSClient()

    ws: WSClient = st.session_state["ws_client"]

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("🟢 Démarrer stream", disabled=st.session_state["ws_running"],
                     use_container_width=True):
            st.session_state["ws_running"]    = True
            st.session_state["ws_detections"] = []
            st.session_state["ws_error"]      = None

            def on_msg(data: dict):
                if data.get("type") == "detection":
                    st.session_state["ws_detections"].append(data)
                    st.session_state["ws_frame_count"] = data.get("frame", 0)

            def on_done():
                st.session_state["ws_running"] = False

            def on_error(err: str):
                st.session_state["ws_running"] = False
                st.session_state["ws_error"]   = err

            ws.start_detection_stream(video_id, model, query, fps, on_msg, on_done, on_error)
            st.rerun()

    with col_stop:
        if st.button("🔴 Arrêter", disabled=not st.session_state["ws_running"],
                     use_container_width=True):
            ws.stop()
            st.session_state["ws_running"] = False
            st.rerun()

    # Statut
    if st.session_state["ws_running"]:
        st.info(f"🔴 Stream actif — frame : {st.session_state['ws_frame_count']}")
    if st.session_state.get("ws_error"):
        st.error(f"Erreur WS : {st.session_state['ws_error']}")

    dets = st.session_state["ws_detections"]
    if dets:
        st.json(dets[-1])
        st.caption(f"{len(dets)} paquets reçus")

    return dets
