@echo off
echo.
echo  Video VLM Platform — Demarrage
echo  ================================
echo.

docker info >nul 2>&1
if errorlevel 1 (
    echo  [ERREUR] Docker Desktop n'est pas lance.
    echo  Ouvrez Docker Desktop puis relancez ce script.
    pause
    exit /b 1
)

if not exist .env (
    copy .env.example .env
    echo  [OK] .env cree depuis .env.example
    echo.
)

echo  Choisissez le mode de demarrage :
echo.
echo  [1] Avec GPU  (RTX 5090 - necessite NVIDIA Container Toolkit dans WSL2)
echo  [2] Sans GPU  (CPU seulement - fonctionne partout, modeles en mode mock)
echo.
set /p choice="Votre choix (1 ou 2) : "

if "%choice%"=="1" (
    echo.
    echo  Demarrage avec GPU...
    docker compose up -d
) else (
    echo.
    echo  Demarrage sans GPU (mode CPU)...
    docker compose -f docker-compose.no-gpu.yml up -d
)

if errorlevel 1 (
    echo.
    echo  [ERREUR] Echec du demarrage. Consultez les logs :
    echo  docker compose logs
    pause
    exit /b 1
)

echo.
echo  ================================
echo  Services demarres !
echo.
echo  Streamlit  : http://localhost:8501
echo  API docs   : http://localhost:8000/docs
echo  Flower     : http://localhost:5555
echo  MinIO      : http://localhost:9001
echo  ================================
echo.
echo  Pour voir les logs : docker compose logs -f
echo  Pour arreter       : docker compose down
echo.
pause
