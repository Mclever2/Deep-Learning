import os

base_path = r"dataset"

contador_mateo = 1
contador_pier = 1

for root, dirs, files in os.walk(base_path):
    for nombre_archivo in files:
   
        if not nombre_archivo.lower().endswith(".mp4"):
            continue

        ruta_vieja = os.path.join(root, nombre_archivo)
        extension = os.path.splitext(nombre_archivo)[1]

   
        if nombre_archivo.upper().startswith("VID_"):
            nuevo_nombre_archivo = f"HOLA_PIER_{contador_pier}{extension}"
            contador_pier += 1
        elif nombre_archivo.startswith("Video de WhatsApp"):
            nuevo_nombre_archivo = f"HOLA_MATEO_{contador_mateo}{extension}"
            contador_mateo += 1
        else:
    
            print(f"Ignorado: {ruta_vieja}")
            continue

      
        ruta_nueva = os.path.join(root, nuevo_nombre_archivo)

 
        os.rename(ruta_vieja, ruta_nueva)
        print(f"Renombrado: {ruta_vieja} → {ruta_nueva}")

print("Renombrado completo.")
