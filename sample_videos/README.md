# Vidéos de démonstration

> ⚠️ **Les fichiers `.mp4` ne sont pas versionnés.** Ils pèsent ~770 Mo au total et
> sont exclus par `.gitignore`. Après un `git clone`, ce dossier est vide : la page
> « 🎬 Démo détection » affiche alors la grille des démos avec les boutons désactivés.
> Deux options pour la peupler — voir [Se procurer les vidéos](#se-procurer-les-vidéos).

L'interface attend exactement ces noms de fichiers dans `sample_videos/`
(définis dans `streamlit_app/pages/1_demo_detection.py`, exposés par l'API sur
`GET /demo-datasets`) :

| # | Fichier attendu | Démo | Spécialité | Requêtes suggérées |
|---|-----------------|------|------------|--------------------|
| 1 | `urban_traffic.mp4` | 🚗 Trafic urbain dense | véhicules & transport | car, truck, bus, person, traffic light, motorcycle, sign |
| 2 | `street_pedestrians.mp4` | 🚶 Piétons & foule | personnes & actions | person, backpack, handbag, umbrella, stroller, dog |
| 3 | `bustling_street.mp4` | 🏙️ Rue commerçante | scène urbaine complète | person, car, bicycle, bus, traffic light, sign, storefront |
| 4 | `aerial_drone_view.mp4` | 🛩️ Vue aérienne drone | géospatial & drone | car, vehicle, person, building, traffic light, bus, motorcycle |
| 5 | `wildlife_safari.mp4` | 🦁 Faune & animaux | animaux & biodiversité | elephant, giraffe, zebra, bird, tree, grass, animal |
| 6 | `team_sports_match.mp4` | ⚽ Sports collectifs | sports & action | person, ball, player, goal, net, crowd, uniform |
| 7 | `night_drive.mp4` | 🌙 Conduite de nuit | conduite & faible luminosité | car, traffic light, sign, person, street light, building, headlight |
| 8 | `market_stalls.mp4` | 🧺 Marché & étals | retail & marchés | person, fruit, vegetable, box, table, bottle, canopy |
| 9 | `construction_site.mp4` | 🏗️ Chantier BTP | industrie & construction | person, truck, crane, hard hat, tool, concrete, excavator |
| 10 | `underwater_dive.mp4` | 🐠 Sous-marin & plongée | sous-marin & océan | fish, coral, diver, turtle, jellyfish, reef, sea |

## Se procurer les vidéos

**Option 1 — n'importe quelle vidéo.** L'onglet « 📁 Uploader une vidéo » de la page
démo accepte MP4, MOV, AVI, MKV, WEBM (2 Go max). Aucune vidéo de démo n'est requise.

**Option 2 — reconstituer le jeu de démo.** Les vidéos d'origine viennent de
[Pexels](https://www.pexels.com/videos/) (libres de droits, sans attribution requise).
Télécharger une vidéo par ligne du tableau, la renommer avec le nom de fichier attendu
et la déposer dans ce dossier. Les noms doivent correspondre exactement — c'est le seul
critère de détection. Le dossier est monté dans le conteneur Streamlit
(`./sample_videos:/app/sample_videos`), un simple dépôt de fichier suffit, sans rebuild.

## Notes d'usage

- **Test le plus rapide** : `street_pedestrians.mp4` (~2 Mo, ~6 s, 1280×720) — c'est la
  vidéo utilisée pour les contrôles end-to-end.
- **Démo la plus riche** : `bustling_street.mp4` ou `market_stalls.mp4` (beaucoup de
  classes simultanées).
- **Vues aériennes** : sur `aerial_drone_view.mp4`, seul `yolo_visdrone` (fine-tuné
  VisDrone2019) donne des résultats exploitables. Les modèles génériques (YOLO-World,
  GroundingDINO) sont entraînés au niveau du sol et détectent très peu en vue plongeante.

## Test en ligne de commande

```bash
curl -X POST "http://localhost:8000/upload" -F "file=@street_pedestrians.mp4"
# -> {"video_id": "...", "duration_sec": 6.34, "fps": 29.97, "width": 1280, ...}
```
