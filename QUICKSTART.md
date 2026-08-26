# 🚀 Guide de démarrage rapide - VLM Demo

## Accès à l'interface

**URL principale** : http://localhost:8501

## Comment utiliser la démo

### 1. Charger une vidéo de démo (recommandé)

1. Ouvrir http://localhost:8501
2. Cliquer sur "🎬 Démo détection"
3. Dans l'onglet **"⚡ Charger une vidéo de démo"**, choisir :
   - 🚶 **Piétons traversant** (1.9 MB) - Test rapide de détection de personnes
   - 🚗 **Trafic urbain** (15 MB) - Détection multi-objets (voitures, feux, panneaux)
   - 🏙️ **Rue animée** (94 MB) - Démo complète avec tous types d'éléments

4. Cliquer sur "🚀 Charger" pour la vidéo choisie
5. Les requêtes de détection recommandées sont automatiquement suggérées !

### 2. Uploader votre propre vidéo

1. Ouvrir http://localhost:8501
2. Cliquer sur "🎬 Démo détection"
3. Dans l'onglet **"📁 Uploader une vidéo"**, glisser-déposer votre fichier
4. Formats supportés : MP4, MOV, AVI, MKV, WEBM (max 2 Go)

### 3. Configurer la détection

Une fois la vidéo chargée :

1. **Utiliser les requêtes recommandées** (si vidéo de démo)
   - Cliquer sur "✅ Utiliser les requêtes recommandées"
   
2. **Ou choisir un preset** :
   - 🚗 **Véhicules** : car, truck, bicycle, motorcycle, bus, traffic light, stop sign
   - 👥 **Personnes** : person, crowd, face, hand, body
   - 📝 **Textes** : sign, text, label, plate, banner
   - 🎯 **Personnalisé** : vos propres requêtes

3. **Ou saisir des requêtes personnalisées** :
   - Exemples : "red car", "person wearing hat", "traffic sign", "bicycle lane"

### 4. Lancer l'analyse

1. Ajuster le seuil de confiance (défaut : 0.35)
2. Activer/désactiver le tracking inter-frames
3. Cliquer sur **"🔍 Lancer la détection"**
4. Attendre la fin de l'analyse (barre de progression)

### 5. Visualiser les résultats

Une fois l'analyse terminée :

- **Métriques globales** : nombre de détections, classes uniques, confiance moyenne
- **Timeline interactive** : graphique temporel des détections
- **Top classes détectées** : les éléments les plus détectés
- **Player vidéo avec overlay** : visualiser les bounding boxes sur la vidéo
- **Export** : télécharger les résultats en CSV ou JSON

## Fonctionnalités avancées

### Page "📊 Analyses détaillées"

Accédez à des visualisations avancées :
- Distribution par classe (graphique)
- Timeline des détections (scatter plot)
- Tableau filtrable des résultats
- Tracking des objets dans le temps
- Clustering de scènes
- Reconnaissance d'actions

### Page "📋 Rapport intelligent"

Générez un rapport synthétique :
- Résumé automatique des analyses
- Statistiques globales
- Export Markdown/JSON
- Configuration personnalisable (style, langue, sections)

## Exemples de requêtes de détection

### Pour la vidéo "Piétons traversant"
```
person
walking
crossing
```

### Pour la vidéo "Trafic urbain"
```
car
truck
traffic light
stop sign
person
bicycle
```

### Pour la vidéo "Rue animée"
```
person
car
bicycle
bus
sign
traffic light
motorcycle
truck
```

### Requêtes avancées (descriptives)
```
red car
person wearing red shirt
white van
yellow taxi
traffic sign
pedestrian crossing
```

## Dépannage

### L'API n'est pas connectée
```bash
# Vérifier le statut des containers
docker ps

# Redémarrer l'API
docker restart video-vlm-platform-api-1
```

### Les workers ne sont pas actifs
```bash
# Vérifier les logs des workers
docker logs video-vlm-platform-worker_vlm-1 --tail 50
docker logs video-vlm-platform-worker_detection-1 --tail 50
```

### Erreur lors de l'upload
- Vérifier la taille du fichier (max 2 Go)
- Vérifier le format (MP4, MOV, AVI, MKV, WEBM)
- Vérifier que l'API est accessible : http://localhost:8000/health

### Aucun résultat de détection
- Ajuster le seuil de confiance (plus bas : 0.2)
- Vérifier que les requêtes sont en anglais
- Essayer des requêtes plus simples : "person", "car"

## Architecture

```
Streamlit (8501) ──► FastAPI (8000) ──► Celery Workers (GPU)
                          │
                    Redis / Qdrant / PostgreSQL / MinIO
```

### Services
- **Streamlit** : Interface utilisateur (port 8501)
- **FastAPI** : API REST (port 8000, docs : http://localhost:8000/docs)
- **Workers Celery** : Traitement GPU (3 workers)
- **Redis** : Broker de messages (port 6379)
- **Qdrant** : Base vectorielle (port 6333)
- **PostgreSQL** : Base de données (port 5432)
- **MinIO** : Stockage objets (port 9000, console : http://localhost:9001)
- **Flower** : Monitoring Celery (port 5555)

## Commandes utiles

```bash
# Démarrer tous les services
make up

# Arrêter tous les services
make down

# Voir les logs
docker compose logs -f

# Redémarrer un service spécifique
docker restart video-vlm-platform-streamlit-1

# Vérifier le statut GPU
docker exec video-vlm-platform-worker_vlm-1 nvidia-smi

# Accéder au shell d'un container
docker exec -it video-vlm-platform-api-1 bash
```

## Ressources

- **Documentation API** : http://localhost:8000/docs
- **Monitoring Celery** : http://localhost:5555
- **Console MinIO** : http://localhost:9001 (minioadmin/minioadmin)

## Vidéos de démo

Les vidéos de démo sont stockées dans `sample_videos/` :
- `street_pedestrians.mp4` (1.9 MB)
- `urban_traffic.mp4` (15 MB)
- `bustling_street.mp4` (94 MB)

Source : Pexels (libre de droits)

---

**Bon test ! 🎯**
