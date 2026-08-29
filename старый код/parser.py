import numpy as np

def parse_pdb_backbone(file_path):
    """
    Читает PDB файл реальной ДНК и вытаскивает координаты атомов фосфора (P),
    которые образуют "хребет" (backbone) нити ДНК.
    """
    coordinates = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Нам нужен только скелет нити, например атомы фосфора 'P'
                atom_name = line[12:16].strip()
                if atom_name == 'P': 
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coordinates.append([x, y, z])
                    
    return np.array(coordinates)




import os
import requests
import numpy as np
from Bio.PDB import PDBParser

def download_pdb_file(pdb_id, output_dir="data/raw"):
    """Скачивает реальный 3D-файл молекулы из Protein Data Bank"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    
    if not os.path.exists(file_path):
        print(f"Скачиваю PDB структуру {pdb_id}...")
        url = f"https://rcsb.org{pdb_id.upper()}.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            raise FileNotFoundError(f"Не удалось скачать PDB с ID: {pdb_id}")
    return file_path

def get_dna_backbone_coordinates(pdb_path):
    """
    Парсит PDB-структуру и вытаскивает 3D-координаты атомов Фосфора (P),
    которые формируют физическую нить ДНК.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("DNA_molecule", pdb_path)
    
    coordinates = []
    
    # Обходим всю молекулу: Модели -> Цепи -> Резидуумы (нуклеотиды) -> Атомы
    for model in structure:
        for chain in model:
            for residue in chain:
                # Проверяем, что это нуклеотид ДНК (обычно DA, DT, DC, DG)
                if residue.get_resname().strip() in ['DA', 'DT', 'DC', 'DG', 'A', 'T', 'C', 'G']:
                    # Нам нужен атом Фосфора 'P' — он связывает нуклеотиды в цепь
                    if 'P' in residue:
                        atom = residue['P']
                        coordinates.append(atom.get_coord())
                        
    return np.array(coordinates)

# --- БЫСТРЫЙ ТЕСТ ОПЫТА ---
if __name__ == "__main__":
    # ID '1BNA' - это классическая спираль ДНК Уотсона-Крика
    # Можно взять '6Z7V' или другие структуры, где ДНК сильно искривлена/завязана
    PDB_ID = "1bna" 
    
    try:
        file_path = download_pdb_file(PDB_ID)
        points_3d = get_dna_backbone_coordinates(file_path)
        
        print(f"\n УСПЕХ! Мы получили реальные физические координаты ДНК.")
        print(f"Всего точек (атомов фосфора в цепи): {points_3d.shape[0]}")
        print(f"Форма массива: {points_3d.shape} (X, Y, Z для каждой точки)")
        print("\nПервые 3 точки для твоего фреймворка:")
        print(points_3d[:3])
        
    except Exception as e:
        print(f"Ошибка: {e}")
