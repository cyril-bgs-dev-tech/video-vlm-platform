-- ═══════════════════════════════════════════════════════════════
-- api/db/init.sql — Schéma PostgreSQL initial
-- Exécuté automatiquement au premier démarrage du container postgres
-- ═══════════════════════════════════════════════════════════════

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- recherche textuelle

-- ── Table des vidéos uploadées ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS videos (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename     TEXT        NOT NULL,
    original_name TEXT       NOT NULL,
    file_path    TEXT        NOT NULL,
    size_bytes   BIGINT,
    duration_sec FLOAT,
    fps          FLOAT,
    width        INTEGER,
    height       INTEGER,
    codec        TEXT,
    status       TEXT        NOT NULL DEFAULT 'uploaded',
    -- status: uploaded | processing | ready | error
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_videos_status     ON videos (status);
CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos (created_at DESC);

-- ── Table des tâches d'analyse ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_tasks (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id     UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    celery_task_id TEXT      UNIQUE,
    task_type    TEXT        NOT NULL,
    -- task_type: sam2 | vlm_query | clustering | detection | ocr | reid | actions | summary | search_index
    status       TEXT        NOT NULL DEFAULT 'pending',
    -- status: pending | running | success | failure | revoked
    params       JSONB       NOT NULL DEFAULT '{}',
    result       JSONB,
    error_msg    TEXT,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tasks_video_id     ON analysis_tasks (video_id);
CREATE INDEX IF NOT EXISTS idx_tasks_celery_id    ON analysis_tasks (celery_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status       ON analysis_tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_type         ON analysis_tasks (task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at   ON analysis_tasks (created_at DESC);

-- ── Table des détections ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS detections (
    id           BIGSERIAL   PRIMARY KEY,
    video_id     UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    task_id      UUID        REFERENCES analysis_tasks(id) ON DELETE SET NULL,
    frame_idx    INTEGER     NOT NULL,
    timestamp_s  FLOAT       NOT NULL,
    class_name   TEXT        NOT NULL,
    confidence   FLOAT       NOT NULL,
    bbox_x1      INTEGER,
    bbox_y1      INTEGER,
    bbox_x2      INTEGER,
    bbox_y2      INTEGER,
    track_id     INTEGER,
    model_name   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_det_video_frame   ON detections (video_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_det_video_class   ON detections (video_id, class_name);
CREATE INDEX IF NOT EXISTS idx_det_track_id      ON detections (video_id, track_id) WHERE track_id IS NOT NULL;

-- ── Table des textes OCR ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ocr_results (
    id           BIGSERIAL   PRIMARY KEY,
    video_id     UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    task_id      UUID        REFERENCES analysis_tasks(id) ON DELETE SET NULL,
    frame_idx    INTEGER     NOT NULL,
    timestamp_s  FLOAT       NOT NULL,
    text_content TEXT        NOT NULL,
    confidence   FLOAT,
    region       TEXT,
    lang         TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ocr_video_id      ON ocr_results (video_id);
CREATE INDEX IF NOT EXISTS idx_ocr_text_trgm     ON ocr_results USING GIN (text_content gin_trgm_ops);

-- ── Table des scènes (clustering) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scenes (
    id              BIGSERIAL   PRIMARY KEY,
    video_id        UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    task_id         UUID        REFERENCES analysis_tasks(id) ON DELETE SET NULL,
    cluster_id      INTEGER     NOT NULL,
    cluster_label   TEXT,
    frame_idx       INTEGER     NOT NULL,
    timestamp_s     FLOAT       NOT NULL,
    umap_x          FLOAT,
    umap_y          FLOAT,
    clip_model      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scenes_video_cluster ON scenes (video_id, cluster_id);

-- ── Table des actions reconnues ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS actions (
    id              BIGSERIAL   PRIMARY KEY,
    video_id        UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    task_id         UUID        REFERENCES analysis_tasks(id) ON DELETE SET NULL,
    timestamp_s     FLOAT       NOT NULL,
    window_end_s    FLOAT,
    action_label    TEXT        NOT NULL,
    probability     FLOAT       NOT NULL,
    model_name      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_actions_video_id    ON actions (video_id);
CREATE INDEX IF NOT EXISTS idx_actions_timestamp   ON actions (video_id, timestamp_s);

-- ── Table des rapports générés ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reports (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id     UUID        NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    task_id      UUID        REFERENCES analysis_tasks(id) ON DELETE SET NULL,
    model_name   TEXT        NOT NULL,
    language     TEXT        NOT NULL DEFAULT 'Français',
    style        TEXT        NOT NULL,
    content_md   TEXT        NOT NULL,
    content_json JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_video_id  ON reports (video_id);

-- ── Trigger : mise à jour automatique de updated_at sur videos ───────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_videos_updated_at ON videos;
CREATE TRIGGER trg_videos_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── Vue pratique : résumé par vidéo ──────────────────────────────────────────
CREATE OR REPLACE VIEW video_summary AS
SELECT
    v.id            AS video_id,
    v.original_name,
    v.duration_sec,
    v.status,
    COUNT(DISTINCT d.id)  FILTER (WHERE d.id IS NOT NULL)  AS n_detections,
    COUNT(DISTINCT o.id)  FILTER (WHERE o.id IS NOT NULL)  AS n_ocr_texts,
    COUNT(DISTINCT s.id)  FILTER (WHERE s.id IS NOT NULL)  AS n_scene_frames,
    COUNT(DISTINCT a.id)  FILTER (WHERE a.id IS NOT NULL)  AS n_actions,
    COUNT(DISTINCT r.id)  FILTER (WHERE r.id IS NOT NULL)  AS n_reports,
    v.created_at
FROM videos v
LEFT JOIN detections    d ON d.video_id = v.id
LEFT JOIN ocr_results   o ON o.video_id = v.id
LEFT JOIN scenes        s ON s.video_id = v.id
LEFT JOIN actions       a ON a.video_id = v.id
LEFT JOIN reports       r ON r.video_id = v.id
GROUP BY v.id, v.original_name, v.duration_sec, v.status, v.created_at;

-- ── Données de démo (optionnel) ───────────────────────────────────────────────
-- INSERT INTO videos (filename, original_name, file_path, status) VALUES
-- ('demo.mp4', 'demo_video.mp4', '/data/videos/demo.mp4', 'ready');
