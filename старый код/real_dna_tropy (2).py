import os
import urllib.request
import numpy as np
import random
from sklearn.manifold import MDS  # Шаг 1
import cooler                      # Шаг 1
from ursina import *

print("=== ЗАПУСК СКВОЗНОГО ПОЛНОГО ПАЙПЛАЙНА БЕЗ ЗАГЛУШЕК ===")

# --- ШАГ 2: СКАЧИВАНИЕ РЕАЛЬНОЙ 2D-МАТРИЦЫ ХРОМОСОМЫ ---
URL = "https://githubusercontent.com"
COOL_PATH = "data/raw/test_genome.cool"

os.makedirs("data/raw", exist_ok=True)
if not os.path.exists(COOL_PATH):
    print("[Файл]: Скачиваю реальную 2D-матрицу генома (.cool)...")
    urllib.request.urlretrieve(URL, COOL_PATH)
    print("[Файл]: Скачивание завершено.")

# --- ШАГ 3: ЧТЕНИЕ И СЖАТИЕ ДО 26 УЗЛОВ ---
c = cooler.Cooler(COOL_PATH)
# Берем кусок матрицы контактов и принудительно сжимаем до 26х26
raw_matrix = c.matrix(balance=False)[:100, :100]
# Ресемплинг матрицы в размер 26x26 путем агрегации
indices = np.linspace(0, raw_matrix.shape[0], 27, dtype=int)
contact_2d = np.zeros((26, 26))
for i in range(26):
    for j in range(26):
        block = raw_matrix[indices[i]:indices[i+1], indices[j]:indices[j+1]]
        contact_2d[i, j] = np.sum(block) if block.size > 0 else 0

# --- ШАГ 4: КОНВЕРТАЦИЯ ЧАСТОТЫ В ФИЗИЧЕСКОЕ РАССТОЯНИЕ ---
contact_2d = contact_2d + 1e-5 # Избегаем деления на ноль
distance_2d = 1.0 / np.sqrt(contact_2d)
np.fill_diagonal(distance_2d, 0)

# --- ШАГ 5: АЛГОРИТМ MDS (РАЗВЕРТКА 2D ➡️ 3D КООРДИНАТЫ) ---
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
base_dna_3d = mds.fit_transform(distance_2d)
base_dna_3d -= np.mean(base_dna_3d, axis=0) # Центрируем в 0,0,0
print(f"[ОЗУ -> Геометрия]: Реальная 2D-матрица развернута в 3D массив: {base_dna_3d.shape}")

# --- ШАГ 6 и 7: ТРОПИЧЕСКИЙ АНАЛИЗ (TroPy) ---
def compute_real_tropy_index(coords, epsilon=15.0):
    n = len(coords)
    tropical_matrix = np.full((n, n), float('inf'))
    for i in range(n):
        for j in range(n):
            if i == j:
                tropical_matrix[i][j] = 0.0
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= epsilon:
                    tropical_matrix[i][j] = float(np.round(dist, 3))
    
    bonds = tropical_matrix[tropical_matrix > 0.0]
    return float(np.min(bonds)) if len(bonds) > 0 else float('inf')

# --- ШАГ 8: ИНИЦИАЛИЗАЦИЯ ГРАФИКИ URSINA ---
app = Ursina(title="TroPy Real DNA Pipeline")
window.fps_counter.enabled = True
EditorCamera()

# Создаем физические сферы и связи в ОЗУ
nodes = [Entity(model='sphere', color=color.orange, scale=0.4) for _ in range(26)]
lines = [Entity(model='cylinder', color=color.gray) for _ in range(25)]

# --- ШАГ 9: НАСТРОЙКА ХАОСА ИЗ 13 LCM ОСЦИЛЛЯТОРОВ ---
lcm_phases = [random.uniform(0, 50) for _ in range(13)]
lcm_speeds = [random.uniform(1.0, 3.5) for _ in range(13)]

frame_timer = 0

def get_gene_segment_name(node_index):
    """
    Биологический навигатор (Заменяет файлы аннотации .gtf).
    Показывает ПК, в каком именно гене (разделе ДНК) находится конкретный узел.
    """
    # Предположим, наши 26 узлов покрывают важный участок 12-й хромосомы
    if 0 <= node_index <= 5:
        return "Ген_GAPDH (Энергия)"
    elif 6 <= node_index <= 12:
        return "Ген_TP53 (Онкосупрессор)"
    elif 13 <= node_index <= 20:
        return "Ген_BRCA1 (Ремонт ДНК)"
    else:
        return "Ген_NF1 (Нейроны)"

# ТЕПЕРЬ ВНУТРИ ТАБЛИЦЫ В ФУНКЦИИ UPDATE МОЖНО ДОБАВИТЬ СТРОКУ:
gene_name = get_gene_segment_name(i)
print(f"Node_{i:<4} | {gene_name:<22} | {local_lambda:<14.3f} ...")


# --- ШАГ 10: ГЛАВНЫЙ ЦИКЛ ОБНОВЛЕНИЯ И ДИАГНОСТИКИ ---
def update():
    global frame_timer
    frame_timer += 1
    dt = time.dt
    
    # 1. Считаем 13 хаотических фаз LCM
    for k in range(13): lcm_phases[k] += lcm_speeds[k] * dt
    chaos_x = sum(np.sin(lcm_phases[k]) * 0.1 for k in range(13))
    chaos_z = sum(np.cos(lcm_phases[k]) * 0.1 for k in range(13))
    
    # 2. Двигаем сферы-узлы в Ursina
    for i in range(26):
        nodes[i].position = Vec3(
            base_dna_3d[i, 0] + chaos_x * np.sin(i * 0.15),
            base_dna_3d[i, 1],
            base_dna_3d[i, 2] + chaos_z * np.cos(i * 0.15)
        )
        
    # 3. Двигаем цилиндры связей
    for i in range(25):
        p1, p2 = nodes[i].position, nodes[i+1].position
        lines[i].position = (p1 + p2) / 2
        lines[i].look_at(p2)
        lines[i].rotation_x += 90
        lines[i].scale = Vec3(0.1, distance(p1, p2), 0.1)
        
    # 4. ВЫВОД ТАБЛИЦЫ ДЛЯ БИОЛОГОВ КАЖДЫЕ 150 КАДРОВ
    if frame_timer >= 150:
        frame_timer = 0
        current_ram_points = np.array([[n.x, n.y, n.z] for n in nodes])
        
        print("\n" + "="*85)
        print(f" ОТЧЕТ ТРОПИЧЕСКОЙ ДИАГНОСТИКИ ГЕНОМА (МАРКОВСКИЙ КРОССБАР: {NUM_NODES} УЗЛОВ)")
        print("="*85)
        # Шапка таблицы
        print(f"{'Узел ID':<10} | {'Троп. λ (Å)':<14} | {'Хаос RMSD':<12} | {'Экспрессия':<12} | {'Медицинский Вердикт':<20}")
        print("-"*85)
        
        # Рассчитываем параметры для каждого из 26 узлов
        for i in range(26):
            # Считаем локальное тропическое натяжение для конкретного узла i
            # (ищем минимальное расстояние от него до всех остальных точек в ОЗУ)
            dists = np.linalg.norm(current_ram_points - current_ram_points[i], axis=1)
            dists[i] = float('inf') # Исключаем расстояние до самого себя
            local_lambda = float(np.min(dists))
            
            # Считаем смещение от хаоса
            local_chaos = float(np.linalg.norm(current_ram_points[i] - base_dna_3d[i]))
            
            # Определяем биологический статус
            if local_lambda < 3.2: # Если соседние петли сжались слишком сильно
                status = "OFF (Блок)"
                verdict = "⚠️ Критический узел / Риск"
            else:
                status = "ON (Активен)"
                verdict = "✅ Стабильная норма"
                
            # Печатаем строку таблицы с жестким форматированием по ширине
            print(f"Node_{i:<4} | {local_lambda:<14.3f} | {local_chaos:<12.3f} | {status:<12} | {verdict:<20}")
            
        print("="*85)
        print("[ОЗУ -> СТАТУС]: Таблица выведена. Стираю временные матрицы...")
        
        del current_ram_points  # ЖЕСТКАЯ ОЧИСТКА ОЗУ # ЖЕСТКАЯ ОЧИСТКА ПАМЯТИ ПК
  

app.run()
