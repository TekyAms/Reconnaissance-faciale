import cv2
import numpy as np
import face_recognition
from datetime import datetime
import time

def charger_noms_images(fichier_classe):
    with open(fichier_classe, 'r') as file:
        lines = file.readlines()

    # Liste pour stocker les chemins des images et les noms des personnes
    image_paths = []
    class_names = []

    # Boucle pour traiter chaque ligne du fichier
    for line in lines:
        line = line.strip().split(',')
        image_path = line[1].strip()
        class_name = line[0].strip()
        image_paths.append(image_path)
        class_names.append(class_name)

    return image_paths, class_names

def charger_images_encodages(image_paths):
    images = [cv2.imread(image_path) for image_path in image_paths]
    
    # Fonction pour trouver les encodages faciaux des images fournies
    def find_encodings(images):
        encode_list = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        return encode_list

    try:
        # Appel de la fonction pour obtenir les encodages des visages connus
        encode_list_known = find_encodings(images)
        print('Encodage complet.')
        return encode_list_known
    except Exception as e:
        print(f"Erreur lors de la génération des encodages : {e}")
        return []

def marquer_presence(personne, status, heure, jour, presence_dict):
    # Vérifier si l'étudiant a déjà été enregistré pour la journée
    if not presence_dict.get(personne, False):
        entree = f"{personne}, {status}, {heure}, {jour}\n"
        try:
            with open('presence.txt', 'r+') as presence_file:
                # Lire toutes les lignes actuelles dans le fichier
                lignes_existantes = presence_file.readlines()

                # Vérifier si l'entrée existe déjà dans le fichier
                existe_deja = any(entree in ligne for ligne in lignes_existantes)

                if not existe_deja:
                    # Ajouter l'entrée uniquement si elle n'existe pas déjà
                    presence_file.write(entree)
                    print(f"Enregistrement réussi : {entree}")
                else:
                    print("Enregistrement déjà existant, ne pas ajouter.")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement de la présence : {e}")

        # Mettre à jour le dictionnaire de présence
        presence_dict[personne] = True

def main():
    fichier_classe = 'classe.txt'

    # Charger les noms et chemins d'images depuis le fichier classe.txt
    image_paths, class_names = charger_noms_images(fichier_classe)

    # Charger les images et effectuer la reconnaissance comme dans votre code
    encode_list_known = charger_images_encodages(image_paths)

    # Initialisation de la capture vidéo à partir de la caméra (indice 0)
    cap = cv2.VideoCapture(0)

    # Initialisation du dictionnaire de présence
    presence_dict = {name: False for name in class_names}

    # Initialisation de la variable status en dehors de la boucle
    status = ""

    try:
        # Boucle principale pour traiter chaque trame vidéo
        while True:
            # Lecture d'une trame vidéo
            _, img = cap.read()

            # Vérification si la trame est vide
            if img is None:
                print("Erreur : Trame vide.")
                break

            # Redimensionnement et conversion des couleurs pour la trame
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Utilisation de face_recognition pour localiser les visages et encoder le visage dans la trame actuelle
            face_current_frame = face_recognition.face_locations(imgS)
            encode_current_frame = face_recognition.face_encodings(imgS, face_current_frame)

            # Boucle pour chaque visage détecté dans la trame
            for face_loc, encode_face in zip(face_current_frame, encode_current_frame):
                # Comparaison des encodages faciaux avec les visages connus pour identifier le visage
                matches = face_recognition.compare_faces(encode_list_known, encode_face)
                face_dis = face_recognition.face_distance(encode_list_known, encode_face)

                # Coordonnées du rectangle autour du visage
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Dessin du rectangle autour du visage et du nom
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                
                # Attribution du nom au visage détecté
                name = "Inconnu"  # Par défaut, le visage est inconnu
                if True in matches:
                    match_index = np.argmin(face_dis)
                    if matches[match_index]:
                        name = class_names[match_index].upper()

                        # Gestion de l'émargement
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        current_day = now.strftime("%Y-%m-%d")

                        # Logique pour déterminer si l'étudiant est en retard, présent, ou absent en fonction de l'heure
                        if current_time < "08:00:00":
                            status = "Present"
                        elif "08:00:00" <= current_time < "12:00:00":
                            status = "Retard"
                        else:
                            status = "Absent"

                        # Marquer la présence dans le fichier
                        marquer_presence(name, status, current_time, current_day, presence_dict)

                # Afficher le nom sur la fenêtre de la caméra
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Affichage de la trame avec les rectangles et le texte
            cv2.imshow('Face Recognition', img)

            # Sortie de la boucle si la touche 'q' est enfoncée
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Ajouter un délai de 100 millisecondes (10 images par seconde)
            time.sleep(0.1)

            # Afficher l'état sur la fenêtre de la caméra (après la boucle pour éviter l'erreur de portée)
            cv2.putText(img, f"Statut: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")

    finally:
        # Libération des ressources vidéo et fermeture des fenêtres
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
