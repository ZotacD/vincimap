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
# 
# - Détecter frame_rate de la vidéo
# 
# - Transformer la vidéo en images en splitant via Min(frame_rate, 12fps) + Vérifier que ce min est % 4 -> frame_rate_splitted
# - stocker dans le dossier "images"
# 
# - Transformer la vidéo en images en splitant via (frame_rate_splitted / 4) 
# - stocker dans le dossier "images_4"
#
# - Appeler colmap et éxécuter automatic_reconstruction via option sparse
#
# - Appeler colmap et éxécuter model_converter pour convertir sparse sous format txt (utile pour le training du modèle splatté)
# - Appeler colmap et éxécuter model_converter pour convertir sparse sous format ply (visualisation nuage de points)
#
# - Lancer simple_trainer avec les options adéquats (des options peuvent être spécifiées par l'utilisateur)
#                                   
# Une fois cela fais, voir pour l'intégration des données Lidar

