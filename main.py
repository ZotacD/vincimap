# TODO: Script complet qui prend une vidéo et qui la transforme en modèle 3D ply.
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
#
# Idée : L'utilisateur choisis des faces (ou itération sur toutes les faces d'un axe), 
# un modèle compare la face avec la vrai photo, récupère l'ID de la photo et récupère les points Lidar associés
# 
# Ensuite, correction des gaussiennes affichées sur le même plan avec les points Lidar associés 