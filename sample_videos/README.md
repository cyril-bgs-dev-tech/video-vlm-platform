# Vidéos de test pour VLM Demo

## Vidéos téléchargées

### 1. street_pedestrians.mp4 (1.9 MB)
**Description** : Piétons traversant une rue  
**Idéal pour** : Détection de personnes, tracking, actions (marche, traversée)  
**Requêtes suggérées** :
```
person
walking
crossing
```

### 2. urban_traffic.mp4 (15 MB)
**Description** : Trafic urbain avec véhicules et intersection  
**Idéal pour** : Détection de véhicules, feux tricolores, panneaux  
**Requêtes suggérées** :
```
car
truck
traffic light
stop sign
person
```

### 3. bustling_street.mp4 (94 MB)
**Description** : Rue urbaine animée avec piétons et véhicules  
**Idéal pour** : Démo complète multi-signaux  
**Requêtes suggérées** :
```
person
car
bicycle
bus
sign
traffic light
```

## Utilisation

### Option 1 : Upload via l'interface
1. Ouvrir http://localhost:8501
2. Aller dans "🎬 Démo détection"
3. Uploader une des vidéos ci-dessus
4. Sélectionner un preset ou des requêtes personnalisées
5. Lancer l'analyse

### Option 2 : Ligne de commande (pour tests rapides)
```bash
# Via l'API directement
curl -X POST "http://localhost:8000/upload" \
  -F "file=@street_pedestrians.mp4"
```

## Caractéristiques des vidéos

| Vidéo | Taille | Durée estimée | Résolution | Contenu principal |
|-------|--------|---------------|------------|-------------------|
| street_pedestrians.mp4 | 1.9 MB | ~5-10s | SD | Piétons |
| urban_traffic.mp4 | 15 MB | ~15-30s | HD | Trafic, véhicules |
| bustling_street.mp4 | 94 MB | ~30-60s | HD | Scène urbaine complète |

## Source

Vidéos libres de droits provenant de [Pexels](https://www.pexels.com/videos/)  
Licence : Utilisation gratuite, pas d'attribution requise

## Recommandations pour la démo

**Pour un test rapide** : Utiliser `street_pedestrians.mp4` (léger, rapide à analyser)

**Pour une démo complète** : Utiliser `bustling_street.mp4` (variété d'éléments à détecter)

**Pour tester le tracking** : Utiliser `urban_traffic.mp4` (véhicules en mouvement)

## Requêtes de détection recommandées

### Preset "Véhicules"
```
car
truck
bicycle
motorcycle
bus
```

### Preset "Personnes"
```
person
crowd
face
```

### Preset "Textes"
```
sign
text
label
plate
```

### Preset "Complet"
```
person
car
bicycle
traffic light
sign
```
