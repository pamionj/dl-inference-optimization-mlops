"""
Script para descargar automáticamente imágenes de frutas desde Wikimedia Commons (Dominio público / Creative Commons)
Útil para inicializar el dataset local de forma rápida y reproducible.
"""

import os
import urllib.request
import urllib.parse
import json
import time

CLASSES = {
    'manzana': 'filetype:bitmap apple fruit',
    'platano': 'filetype:bitmap banana fruit',
    'naranja': 'filetype:bitmap orange fruit citrus',
    'frutilla': 'filetype:bitmap strawberry fruit',
    'uva': 'filetype:bitmap grapes fruit bunch'
}

DATASET_DIR = "data/dataset_frutas"
IMAGES_PER_CLASS = 20

def descargar_imagenes():
    print("=" * 60)
    print("DESCARGADOR DE IMÁGENES DE MUESTRA PARA DATASET DE FRUTAS")
    print("=" * 60)
    
    headers = {
        'User-Agent': 'FruitDatasetFetcher/1.0 (MachineLearningStudent; contacto@inacap.cl)'
    }

    for class_name, query in CLASSES.items():
        folder = os.path.join(DATASET_DIR, class_name)
        os.makedirs(folder, exist_ok=True)
        
        # Verificar cuántas ya existen
        existing = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(existing) >= IMAGES_PER_CLASS:
            print(f"[{class_name}] Ya tiene {len(existing)} imágenes. Saltando...")
            continue
            
        print(f"\nBuscando imágenes para '{class_name}' ({query})...")
        encoded_query = urllib.parse.quote(query)
        api_url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6&gsrsearch={encoded_query}&gsrlimit=30&prop=imageinfo&iiprop=url|mime"
        
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                
            pages = data.get('query', {}).get('pages', {})
            count = len(existing)
            
            for pid, info in pages.items():
                if count >= IMAGES_PER_CLASS:
                    break
                    
                image_infos = info.get('imageinfo', [])
                if not image_infos:
                    continue
                    
                img_url = image_infos[0].get('url')
                mime = image_infos[0].get('mime', '')
                
                # Filtrar solo JPEG o PNG
                if 'jpeg' in mime or 'jpg' in mime:
                    ext = '.jpg'
                elif 'png' in mime:
                    ext = '.png'
                else:
                    continue
                
                dest_path = os.path.join(folder, f"{class_name}_{count+1:03d}{ext}")
                try:
                    img_req = urllib.request.Request(img_url, headers=headers)
                    with urllib.request.urlopen(img_req, timeout=15) as img_resp, open(dest_path, 'wb') as out_f:
                        out_f.write(img_resp.read())
                    count += 1
                    print(f"  [+] Descargada {count}/{IMAGES_PER_CLASS}: {os.path.basename(dest_path)}")
                    time.sleep(0.3)  # Respetar rate limits
                except Exception as e:
                    # Si falla una URL particular, continuar con la siguiente
                    continue
                    
            print(f"[{class_name}] Total finalizado: {count} imágenes.")
        except Exception as e:
            print(f"[ERROR] No se pudo buscar imágenes para '{class_name}': {e}")

    print("\nProceso finalizado. Ejecuta 'python scripts/verify_dataset.py' para auditar los resultados.")

if __name__ == "__main__":
    descargar_imagenes()

