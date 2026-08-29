import os
import requests
import numpy as np
from Bio.PDB import PDBParser

def get_real_dna_backbone(pdb_id, output_dir="data/raw"):
    """Скачивает реальную ДНК и вытаскивает координаты скелета (атомы P)"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    
    if not os.path.exists(file_path):
        print(f"[{pdb_id}] Скачивание структуры из PDB...")
        url = f"https://rcsb.org{pdb_id.upper()}.pdb"
        res = requests.get(url)
        if res.status_code == 200:
            with open(file_path, 'wb') as f: f.write(res.content)
        else:
            raise FileNotFoundError(f"Не удалось найти PDB: {pdb_id}")
            
    # Парсим структуру и собираем 3D-точки
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("DNA", file_path)
    coords = []
    for model in struct:
        for chain in model:
            for residue in chain:
                if 'P' in residue: # Фосфорный хребет ДНК
                    coords.append(residue['P'].get_coord())
    return np.array(coords)

def compress_to_26_nodes(coordinates, target_nodes=26):
    """
    Равномерно сжимает нить ДНК любой длины до строго заданного 
    количества узлов (шарниров) для твоего Марковского графа.
    """
    total_points = len(coordinates)
    if total_points < target_nodes:
        raise ValueError(f"В исходной ДНК всего {total_points} точек. Нужно минимум {target_nodes}.")
    
    # Интерполируем/выбираем индексы так, чтобы равномерно распределить 26 узлов по длине
    indices = np.linspace(0, total_points - 1, target_nodes, dtype=int)
    compressed_coords = coordinates[indices]
    
    return compressed_coords

# --- ПРОВЕРКА ПЕРВОГО ШАГА ---
if __name__ == "__main__":
    # 1BNA - тестовая классическая спираль. 
    # Для реальных узлов потом возьмем структуры ДНК-топоизомераз или аптамеров (например, 6Z7V)
    PDB_TEST_ID = "1bna" 
    
    print("=== ШАГ 1: ПОДГОТОВКА ДАННЫХ ДНК ===")
    raw_points = get_real_dna_backbone(PDB_TEST_ID)
    print(f"Исходная нить ДНК содержит атомов (точек): {len(raw_points)}")
    
    # Сжимаем под твою архитектуру на 26 узлов
    dna_26 = compress_to_26_nodes(raw_points, target_nodes=26)
    print(f"Нить успешно сжата! Размер массива для ОЗУ: {dna_26.shape}")
    print("\nГотовые 3D-координаты первых 3 узлов графа:")
    print(dna_26[:3])
    
    # Сохраняем промежуточный результат, чтобы не забивать ОЗУ
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/dna_26_nodes.npy", dna_26)
    print("\n[Чекпоинт]: Координаты сохранены в data/processed/dna_26_nodes.npy")
