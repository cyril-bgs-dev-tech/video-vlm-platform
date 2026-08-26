# ═══════════════════════════════════════════════════════════
# Makefile — Video VLM Platform
# ═══════════════════════════════════════════════════════════
.PHONY: help up down build logs shell-api shell-streamlit models gpu-check clean

help:
	@echo "Video VLM Platform — Commandes disponibles :"
	@echo ""
	@echo "  make up            Démarrer tous les services"
	@echo "  make down          Arrêter tous les services"
	@echo "  make build         Reconstruire les images Docker"
	@echo "  make logs          Suivre les logs (streamlit + api + workers)"
	@echo "  make gpu-check     Vérifier la visibilité GPU dans Docker"
	@echo "  make models        Télécharger les modèles"
	@echo "  make flower        Ouvrir le dashboard Celery (Flower)"
	@echo "  make shell-api     Shell dans le container API"
	@echo "  make shell-streamlit  Shell dans Streamlit"
	@echo "  make redis-flush   Vider la queue Redis"
	@echo "  make clean         Supprimer les volumes de données"

up:
	docker compose up -d
	@echo "✅ Services démarrés"
	@echo "  Streamlit  : http://localhost:8501"
	@echo "  FastAPI    : http://localhost:8000/docs"
	@echo "  Flower     : http://localhost:5555"
	@echo "  MinIO      : http://localhost:9001"

down:
	docker compose down

build:
	docker compose build --no-cache

rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

logs:
	docker compose logs -f streamlit api worker_vlm worker_detection worker_clustering

logs-api:
	docker compose logs -f api

logs-workers:
	docker compose logs -f worker_vlm worker_detection worker_clustering

gpu-check:
	docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

models:
	chmod +x download_models.sh && ./download_models.sh

flower:
	@echo "Flower disponible sur http://localhost:5555"
	@open http://localhost:5555 2>/dev/null || xdg-open http://localhost:5555 2>/dev/null || true

shell-api:
	docker compose exec api bash

shell-streamlit:
	docker compose exec streamlit bash

shell-vlm-worker:
	docker compose exec worker_vlm bash

redis-flush:
	docker compose exec redis redis-cli FLUSHALL
	@echo "✅ Queue Redis vidée"

redis-inspect:
	docker compose exec redis redis-cli INFO keyspace

celery-inspect:
	docker compose exec api celery -A workers.celery_app inspect active

celery-purge:
	docker compose exec api celery -A workers.celery_app purge -f

db-shell:
	docker compose exec postgres psql -U vlm -d vlmdb

qdrant-collections:
	curl -s http://localhost:6333/collections | python3 -m json.tool

minio-open:
	@open http://localhost:9001 2>/dev/null || xdg-open http://localhost:9001 2>/dev/null || echo "http://localhost:9001 (admin:minioadmin)"

clean:
	docker compose down -v
	@echo "⚠️  Volumes supprimés (données persistantes effacées)"

status:
	docker compose ps
	@echo ""
	@echo "GPU :"
	@docker compose exec worker_vlm nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  worker_vlm non démarré"

test:
	pip install -r tests/requirements-test.txt -q
	pytest tests/ -v --tb=short

test-coverage:
	pip install -r tests/requirements-test.txt -q
	pytest tests/ --cov=api --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --ignore=tests/test_workers.py -k "not slow"

logs-json:
	docker compose logs -f --no-log-prefix api worker_vlm worker_detection worker_clustering | python3 -c "import sys,json; [print(json.dumps(json.loads(l),indent=2)) if l.startswith(\'{\') else print(l) for l in sys.stdin]"

db-init:
	docker compose exec postgres psql -U vlm -d vlmdb -f /docker-entrypoint-initdb.d/init.sql
	@echo "✅ Schéma DB initialisé"

db-reset:
	docker compose exec postgres psql -U vlm -d vlmdb -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-init

minio-init:
	docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
	docker compose exec minio mc mb local/videos --ignore-existing
	@echo "✅ Bucket MinIO créé"

env-check:
	@echo "=== Vérification de l\'environnement ==="
	@docker --version
	@docker compose version
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "⚠️  nvidia-smi non disponible"
	@python3 --version

