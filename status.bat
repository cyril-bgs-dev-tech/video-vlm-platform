@echo off
echo.
echo  === État des services ===
docker compose ps
echo.
echo  === GPU ===
docker compose exec worker_vlm nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>nul || echo  (worker_vlm non démarré)
echo.
pause
