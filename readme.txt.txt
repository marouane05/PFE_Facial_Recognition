

***********************************

1. TakeImageForFDetectorImage.py

(we take images for a person )

-- On stock ces données dans data2/images 

2. AugmentData.py

   A. il Prend les images vc ces fichiers des cordonnées json (crée par labelme) 
   B. il affecte une augmentation
   C. Il l'envoie vers des dossiers augmented_data2

3. BuildPipeline.py

Pour construire le modèle 


4. Recognition folder***************************************************************************************************

4.1 Take_Recognition_Pictures.py 

Il prend 30 images pour chaque nouveau user en proposons un id pour chaque user

4.2.  trainer.py
 
Pour faire le training et détecter les personnes (faces)

4.3. recognizer_Detection_FROM_Scratch.py

Detection from scratch + modèle de reconnaissance 

********************************************************

Graphique Builder :
https://visualtk.com/
