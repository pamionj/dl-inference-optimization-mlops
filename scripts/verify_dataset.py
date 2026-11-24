"""
Script para auditar y verificar el conjunto de datos de imágenes de frutas.
Comprueba estructura de carpetas, balance de clases, formatos y suficiencia para train/val.
"""

import os
from pathlib import Path

def verificar_dataset(dataset_dir="data/dataset_frutas", min_images_per_class=15):
    print("=" * 60)
    print("AUDITORÍA DE DATASET PARA EVALUACIÓN 3 MACHINE LEARNING")
    print("=" * 60)
    
    path = Path(dataset_dir)
    if not path.exists():
        print(f"[ERROR] No se encontró el directorio: {dataset_dir}")
        return False
    
    classes = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not classes:
        print(f"[ERROR] No hay subcarpetas de clases dentro de {dataset_dir}")
        print("Crea carpetas como: manzana, platano, naranja, etc.")
        return False
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    total_images = 0
    class_counts = {}
    corrupted = []
    
    # Intentar importar PIL para validar que abran bien
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pil_available = False
        print("[AVISO] PIL (Pillow) no está instalado; solo se verificará extensión de archivos.")

    print(f"\nDirectorio analizado: {path.resolve()}")
    print(f"Total de clases detectadas: {len(classes)}\n")
    print(f"{'Clase':<20} | {'Imágenes':<10} | {'Train (~80%)':<12} | {'Val (~20%)':<10}")
    print("-" * 60)

    for cls in sorted(classes, key=lambda c: c.name):
        files = [f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        valid_count = 0
        
        for f in files:
            if pil_available:
                try:
                    with Image.open(f) as img:
                        img.verify()
                    valid_count += 1
                except Exception as e:
                    corrupted.append((str(f), str(e)))
            else:
                valid_count += 1
                
        class_counts[cls.name] = valid_count
        total_images += valid_count
        train_est = int(valid_count * 0.8)
        val_est = valid_count - train_est
        print(f"{cls.name:<20} | {valid_count:<10} | {train_est:<12} | {val_est:<10}")

    print("-" * 60)
    print(f"{'TOTAL':<20} | {total_images:<10} | {int(total_images*0.8):<12} | {total_images - int(total_images*0.8):<10}")
    print("=" * 60)

    if corrupted:
        print(f"\n[ALERTA] Se encontraron {len(corrupted)} imágenes dañadas:")
        for file_path, err in corrupted:
            print(f"  - {file_path}: {err}")

    # Evaluación de suficiencia para la rúbrica (Criterio 3.1.3)
    warnings = []
    if total_images == 0:
        warnings.append("El dataset está completamente vacío. Debes añadir imágenes en cada carpeta.")
    else:
        for cls_name, count in class_counts.items():
            if count == 0:
                warnings.append(f"La clase '{cls_name}' no tiene imágenes.")
            elif count < min_images_per_class:
                warnings.append(f"La clase '{cls_name}' tiene solo {count} imágenes. Se recomiendan mínimo {min_images_per_class} para que el 20% de validación no quede vacío.")
        
        val_total = total_images - int(total_images * 0.8)
        if val_total < len(classes):
            warnings.append(f"Con solo {total_images} imágenes en total, la partición de validación tendrá apenas {val_total} imágenes, lo que impedirá calcular la matriz de confusión con precisión.")

    if warnings:
        print("\n[RECOMENDACIONES SEGÚN RÚBRICA - CRITERIO 3.1.3]:")
        for w in warnings:
            print(f"  * {w}")
        return False
    else:
        print("\n[OK] El dataset cumple con los requisitos mínimos de estructura y balance.")
        return True

if __name__ == "__main__":
    verificar_dataset()

