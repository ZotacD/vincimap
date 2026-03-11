# TODO: Script complet qui prend une vidéo et qui la transforme en modèle 3D ply.
#
# ATTENTION : Fork de pycolmap nécessaire pour que simple_trainer.py fonctionne, 
# modifier "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" dasn requirements.txt
# -> Modifier scene_manager.py, _load_images_txt et _load_points3D_txt 
#   map(...) -> list(map(...))
#
# Les étapes suivantes doivent être respectées (en suivant le google doc "Colmap") :
#
# - Vérifier si cuda installé + version demandée
# - Intégrer colmap dans la codebase ou vérifier si colmap installé + version demandée ?
# - Vérifier ffmpeg installé 
#
# - Détecter frame_rate de la vidéo
# 
# - Transformer la vidéo en images via Min(frame_rate, 12fps)
# - Stocker dans le dossier "images"
#
# - Redimensionner les images de "images" par un facteur 4
# - Stocker dans le dossier "images_4" (utile pour le training du modèle splatté avec --data_factor 4)
#
# - Appeler colmap et éxécuter automatic_reconstruction via option sparse
#
# - Appeler colmap et éxécuter model_converter pour convertir sparse sous format txt (utile pour le training du modèle splatté)
# - Appeler colmap et éxécuter model_converter pour convertir sparse sous format ply (visualisation nuage de points)
#
# - Lancer simple_trainer avec les options adéquats (des options peuvent être spécifiées par l'utilisateur)
#                                   
# Une fois cela fais, voir pour l'intégration des données Lidar

