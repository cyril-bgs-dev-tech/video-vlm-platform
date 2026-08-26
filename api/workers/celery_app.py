# ════════════════════════════════════════════════════════════
# workers/celery_app.py
# ═══════════════════════════════════════════════════════════
"""
celery_app.py — Configuration Celery + Redis broker
"""
import os
from celery import Celery
from kombu import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "vlm_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "workers.vlm_worker",
        "workers.detection_worker",
        "workers.clustering_worker",
    ],
)

celery_app.conf.update(
    task_serializer       = "json",
    result_serializer     = "json",
    accept_content        = ["json"],
    result_expires        = 3600,
    worker_prefetch_multiplier = 1,
    task_acks_late        = True,
    task_reject_on_worker_lost = True,
    # solo évite le fork → CUDA utilisable dans le worker
    worker_pool           = os.getenv("CELERY_POOL", "solo"),
    worker_concurrency    = int(os.getenv("CELERY_CONCURRENCY", "1")),
    worker_max_tasks_per_child = int(os.getenv("CELERY_MAX_TASKS_PER_CHILD", "0")) or None,
    task_queues = (
        Queue('celery', routing_key='celery'),
        Queue('vlm_queue', routing_key='vlm_queue'),
        Queue('detection_queue', routing_key='detection_queue'),
        Queue('clustering_queue', routing_key='clustering_queue'),
    ),
    task_default_queue = 'celery',
    task_default_exchange = 'celery',
    task_default_routing_key = 'celery',
)

# Routes par défaut pour les tâches
celery_app.conf.task_routes = {
    'workers.vlm_worker.*': {'queue': 'vlm_queue'},
    'workers.detection_worker.*': {'queue': 'detection_queue'},
    'workers.clustering_worker.*': {'queue': 'clustering_queue'},
}
