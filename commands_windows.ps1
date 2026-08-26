# ═══════════════════════════════════════════════════════════════
# commands_windows.ps1 — Équivalents PowerShell du Makefile
#
# Usage : ouvrir PowerShell dans le dossier du projet
#   . .\commands_windows.ps1   (charger les fonctions)
#   vlm-up                     (démarrer)
#   vlm-down                   (arrêter)
#   vlm-logs                   (logs temps réel)
#   vlm-status                 (état des services)
#   vlm-test                   (lancer les tests)
#   vlm-flush                  (vider la queue Redis)
# ═══════════════════════════════════════════════════════════════

function vlm-up {
    Write-Host "`n  Démarrage de la stack..." -ForegroundColor Cyan
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "  [OK] .env créé depuis .env.example" -ForegroundColor Green
    }
    docker compose up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n  Services démarrés :" -ForegroundColor Green
        Write-Host "  Streamlit  → http://localhost:8501"
        Write-Host "  API docs   → http://localhost:8000/docs"
        Write-Host "  Flower     → http://localhost:5555"
        Write-Host "  MinIO      → http://localhost:9001`n"
    }
}

function vlm-down {
    Write-Host "  Arrêt des services..." -ForegroundColor Yellow
    docker compose down
}

function vlm-restart {
    vlm-down
    vlm-up
}

function vlm-logs {
    docker compose logs -f streamlit api worker_vlm worker_detection worker_clustering
}

function vlm-logs-api {
    docker compose logs -f api
}

function vlm-logs-workers {
    docker compose logs -f worker_vlm worker_detection worker_clustering
}

function vlm-status {
    Write-Host "`n  === Services ===" -ForegroundColor Cyan
    docker compose ps
    Write-Host "`n  === GPU ===" -ForegroundColor Cyan
    docker compose exec worker_vlm nvidia-smi `
        --query-gpu=name,memory.used,memory.total,utilization.gpu `
        --format=csv,noheader 2>$null
    Write-Host ""
}

function vlm-test {
    Write-Host "  Lancement des tests..." -ForegroundColor Cyan
    pip install -r tests/requirements-test.txt -q
    pytest tests/ -v
}

function vlm-test-api {
    pytest tests/test_api.py -v
}

function vlm-test-workers {
    pytest tests/test_workers.py -v
}

function vlm-build {
    docker compose build --no-cache
}

function vlm-flush {
    Write-Host "  Vidage de la queue Redis..." -ForegroundColor Yellow
    docker compose exec redis redis-cli FLUSHALL
    Write-Host "  [OK] Queue Redis vidée"
}

function vlm-shell-api {
    docker compose exec api bash
}

function vlm-shell-streamlit {
    docker compose exec streamlit bash
}

function vlm-db-shell {
    docker compose exec postgres psql -U vlm -d vlmdb
}

function vlm-gpu-check {
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
}

function vlm-open {
    Start-Process "http://localhost:8501"   # Streamlit
    Start-Process "http://localhost:8000/docs"  # API docs
}

function vlm-help {
    Write-Host "`n  Commandes disponibles :" -ForegroundColor Cyan
    Write-Host "  vlm-up          Démarrer tous les services"
    Write-Host "  vlm-down        Arrêter tous les services"
    Write-Host "  vlm-restart     Redémarrer"
    Write-Host "  vlm-logs        Logs temps réel"
    Write-Host "  vlm-logs-api    Logs API uniquement"
    Write-Host "  vlm-logs-workers Logs workers GPU"
    Write-Host "  vlm-status      État des services + GPU"
    Write-Host "  vlm-test        Lancer tous les tests"
    Write-Host "  vlm-test-api    Tests API uniquement"
    Write-Host "  vlm-build       Reconstruire les images"
    Write-Host "  vlm-flush       Vider la queue Redis"
    Write-Host "  vlm-shell-api   Shell dans le container API"
    Write-Host "  vlm-gpu-check   Vérifier visibilité GPU"
    Write-Host "  vlm-open        Ouvrir Streamlit + API docs dans le navigateur"
    Write-Host ""
}

# Afficher l'aide au chargement
vlm-help
