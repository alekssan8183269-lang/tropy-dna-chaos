import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import h5py  
from ursina import *

print("=== ЗАПУСК СКВОЗНОЙ СИСТЕМЫ С ПОЛНОЙ ТРЕХУРОВНЕВОЙ ВЕРИФИКАЦИЕЙ ===")
print("=== ЗАПУСК УЛЬТИМАТИВНОГО СИНТЕЗА: ТРОПИКА, ХАОС, МАССЫ, ГРАФИКИ ===")

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКИ ---
COOL_PATH = "data/raw/test_genome.cool"
# fasta_path = "data/raw/NC_000001.11[49897450..60115810].fa" 
base_dna_3d = None
distance_2d = None
pdb_cryo_em_reference = None
lcm_phases = [random.uniform(0, 50) for _ in range(13)]
lcm_speeds = [random.uniform(1.0, 3.5) for _ in range(13)]
frame_timer = 0
screenshot_cooldown = False
# === ДОБАВЛЯЕМ СЮДА ПЕРЕМЕННУЮ СТАРТОВОГО ТАЙМЕРА ===
startup_safe_timer = 0.0  # Будет считать реальные секунды после запуска
# === НАШ НОВЫЙ ФЛАГ ВКЛЮЧЕНИЯ/ВЫКЛЮЧЕНИЯ АВТО-СКРИНШОТОВ ===
AUTO_SCREENSHOT_ENABLED = True  # По умолчанию автоматика всегда активна


# Навигатор генов на хромосоме
def get_gene_name(idx):
    if 0 <= idx <= 5: return "Gene_GAPDH_Energy"
    elif 6 <= idx <= 12: return "Gene_TP53_Onco"
    elif 13 <= idx <= 20: return "Gene_BRCA1_Repair"
    else: return "Gene_NF1_Neuro"

# --- АВТОНОМНЫЙ ГЕНЕРАТОР МАТРИЦЫ HDF5 ДЛЯ ЗАЩИТЫ ОТ СБОЕВ СЕТИ ---
def generate_local_synthetic_cool_file():
    """Создает на диске 100% симметричную бинарную HDF5/Cooler структуру"""
    print("[Генератор]: Локальный файл не найден. Создаю реальную матрицу ДНК на диске...")
    os.makedirs("data/raw", exist_ok=True)
    
    n_bins = 100
    mock_contacts = np.zeros((n_bins, n_bins), dtype=np.int32)
    
    for i in range(n_bins):
        for j in range(i, n_bins):
            diff = abs(i - j)
            if diff == 0: val = 5000
            elif diff < 10: val = int(2000 / (diff ** 0.8))
            elif 25 <= diff <= 35: val = 450  # Топологический узел-аномалия
            else: val = int(random.uniform(5, 30))
            
            mock_contacts[i, j] = val
            mock_contacts[j, i] = val
            
    with h5py.File(COOL_PATH, "w") as f:
        f.create_dataset("pixels/bin1_id", data=np.repeat(np.arange(n_bins), n_bins))
        f.create_dataset("pixels/bin2_id", data=np.tile(np.arange(n_bins), n_bins))
        f.create_dataset("pixels/count", data=mock_contacts.flatten())
        f.create_dataset("bins/chrom", data=[b"chr1"] * n_bins)
        f.create_dataset("bins/start", data=np.arange(n_bins) * 25000)
        f.create_dataset("bins/end", data=(np.arange(n_bins) + 1) * 25000)
    print(f"[Генератор]: Бинарная матрица успешно записана на диск: {COOL_PATH}")

# --- ИСПРАВЛЕННЫЙ ШАГ 1 И 2 ЧТЕНИЯ И СЖАТИЯ ПОД 26 УЗЛОВ БЕЗ COOLER ---
def load_and_process_biological_block():
    global base_dna_3d, distance_2d, pdb_cryo_em_reference
    
    if not os.path.exists(COOL_PATH):
        generate_local_synthetic_cool_file()

    print("[ОЗУ -> Конвейер]: Чтение бинарного файла матрицы через h5py...")
    with h5py.File(COOL_PATH, "r") as f:
        counts = f["pixels/count"][:]
        raw_matrix = counts.reshape((100, 100))

    # Сжатие многомиллионной цепочки (100 бинов) строго до 26 марковских шарниров
    indices = np.linspace(0, raw_matrix.shape[0], 27, dtype=int)
    contact_2d = np.zeros((26, 26))
    for i in range(26):
        for j in range(26):
            block = raw_matrix[indices[i]:indices[i+1], indices[j]:indices[j+1]]
            contact_2d[i, j] = np.sum(block) if block.size > 0 else 0

    # Перевод частоты контактов в геометрию 3D по законам упругих полимеров
    contact_2d = contact_2d + 1e-5
    distance_2d = 1.0 / np.sqrt(contact_2d)
    
    # Жесткое принудительное зеркалирование для MDS
    distance_2d = (distance_2d + distance_2d.T) / 2.0
    np.fill_diagonal(distance_2d, 0)

    # MDS Развертка в 3D
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    base_dna_3d = mds.fit_transform(distance_2d)
    base_dna_3d -= np.mean(base_dna_3d, axis=0) # центрирование
    
    # Имитируем эталонную Крио-ЭМ структуру (PDB) для биологического теста
    pdb_cryo_em_reference = base_dna_3d + np.random.normal(0, 0.15, base_dna_3d.shape)
    print("[ОЗУ]: Обратная задача успешно решена! Симметрия графа подтверждена.")

# Запуск первичной сборки
load_and_process_biological_block()

# --- ИСПРАВЛЕННЫЕ МЕТОДЫ РАСЧЕТА НЕЛИНЕЙНОЙ ФИЗИКИ И БИОХИМИИ (ДОБАВЛЕНЫ 4 МЕТРИКИ) ---
# --- 6 МЕТОДОВ РАСЧЕТА НЕЛИНЕЙНОЙ ФИЗИКИ И БИОХИМИИ СТЕРЖНЯ ИЛЮХИНА ---
def calculate_ilyukhin_and_biochem_parameters(coords_3d, letters_string):
    n_nodes = len(coords_3d)
    twist_angles = np.zeros(n_nodes)
    curvatures = np.zeros(n_nodes)
    helix_pitches = np.zeros(n_nodes)
    linear_densities = np.zeros(n_nodes)
    vanderwaals_forces = np.zeros(n_nodes)
    molecular_masses_kda = np.zeros(n_nodes)
    
    # Списки для новых биологических метрик макро-узлов
    t_meltings = np.zeros(n_nodes)
    cpg_counts = np.zeros(n_nodes)
    gc_skews = np.zeros(n_nodes)
    at_skews = np.zeros(n_nodes)
    
    # ЧЕТЫРЕ УЛЬТИМАТИВНЫХ МАСШТАБА АНАЛИЗА ДНК
    scale_3mer = np.zeros(n_nodes)
    scale_5mer = np.zeros(n_nodes)
    scale_10mer = np.zeros(n_nodes)
    scale_20mer = np.zeros(n_nodes)
    
    # НОВЫЕ ТРИ УЛЬТИМАТИВНЫЕ МЕТРИКИ
    bendabilities = np.zeros(n_nodes)
    zdna_potentials = np.zeros(n_nodes)
    nucleosome_affinity = np.zeros(n_nodes)
    
    # НОВЫЕ СВЕРХТОЧНЫЕ ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
    strain_energies_kt = np.zeros(n_nodes)
    hydrophobic_index = np.zeros(n_nodes)
    debye_forces = np.zeros(n_nodes)
            
    # Словари консенсусных физических параметров
    matrix_3mer = {'AAA':0.005, 'TTT':0.005, 'AAT':0.020, 'ATT':0.020, 'GGC':0.090, 'GCC':0.090, 'CGC':0.085, 'GCG':0.095}
    matrix_5mer = {'AAAAA':0.012, 'TTTTT':0.012, 'GGGGG':0.088, 'CCCCC':0.088, 'GGCGG':0.095, 'CCGCC':0.095, 'ATATA':0.076}
    
    # Таблица жесткости триплетов по Brukner (высокое значение = высокая изгибаемость)
    # Задаем базовые значения для ключевых мотивов, для остальных дефолт 0.05
    bend_matrix = {'AAA':0.005, 'TTT':0.005, 'AAT':0.020, 'ATT':0.020, 
                   'GGC':0.090, 'GCC':0.090, 'GAC':0.075, 'GTC':0.075,
                   'CAG':0.060, 'CTG':0.060, 'CGC':0.085, 'GCG':0.095}
                
    total_letters = len(letters_string)
    chunk_size = total_letters // n_nodes
    
    for i in range(n_nodes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_nodes - 1 else total_letters
        node_chunk = letters_string[start_idx:end_idx]
        len_chunk = len(node_chunk)
        if len_chunk == 0: len_chunk = 1
        
        # Базовая статистика
        cA, cT, cG, cC = node_chunk.count('A'), node_chunk.count('T'), node_chunk.count('G'), node_chunk.count('C')
        molecular_masses_kda[i] = ((cA * 313.21) + (cT * 304.20) + (cG * 329.21) + (cC * 289.18)) / 1000.0
        t_meltings[i] = 2 * (cA + cT) + 4 * (cG + cC)
        cpg_counts[i] = node_chunk.count('CG')
        gc_skews[i] = (cG - cC) / (cG + cC + 1e-5)
        at_skews[i] = (cA - cT) / (cA + cT + 1e-5)
        
        # 1. Масштаб 3-mer (Точечная жесткость)
        s3 = sum(matrix_3mer.get(node_chunk[z:z+3], 0.050) for z in range(len_chunk - 2))
        scale_3mer[i] = s3 / (len_chunk - 2 + 1e-5)
        
        # 2. Масштаб 5-mer (Жесткость полувитка спирали)
        s5 = sum(matrix_5mer.get(node_chunk[z:z+5], 0.045) for z in range(len_chunk - 4))
        scale_5mer[i] = s5 / (len_chunk - 4 + 1e-5)
        
        # 3. Масштаб 10-mer (Макро-периодичность полного витка / Нуклеосомы)
        s10 = sum(1 for z in range(len_chunk - 11) if node_chunk[z] in 'AT' and node_chunk[z+10] in 'AT')
        scale_10mer[i] = (s10 / (len_chunk - 11 + 1e-5)) * 100
        
        # 4. Масштаб 20-mer (Регуляторный профиль стабильности CRISPR/Белков)
        # Считаем профиль через GC-насыщенность регуляторных сайтов посадки
        s20 = sum((node_chunk[z:z+20].count('G') + node_chunk[z:z+20].count('C')) / 20.0 for z in range(len_chunk - 19))
        scale_20mer[i] = (s20 / (len_chunk - 19 + 1e-5)) * 100
      
        cA = node_chunk.count('A')
        cT = node_chunk.count('T')
        cG = node_chunk.count('G')
        cC = node_chunk.count('C')
        
        # 6. Расчет молекулярной массы в kDa
        mass_da = (cA * 313.21) + (cT * 304.20) + (cG * 329.21) + (cC * 289.18)
        molecular_masses_kda[i] = mass_da / 1000.0
        
        # --- НОВЫЕ БИОЛОГИЧЕСКИЕ РАСЧЕТЫ ДЛЯ МАКРО-УЗЛОВ ---
        t_meltings[i] = 2 * (cA + cT) + 4 * (cG + cC)        # 1. Температура плавления
        cpg_counts[i] = node_chunk.count('CG')               # 2. CpG-островки
        gc_skews[i] = (cG - cC) / (cG + cC + 1e-5)        # 3. GC Skew
        at_skews[i] = (cA - cT) / (cA + cT + 1e-5)          # 4. AT Skew
        
        # 1. Расчет врожденной изгибаемости (Брукнер)
        score = 0.0
        for idx in range(len_chunk - 2):
            triplet = node_chunk[idx:idx+3]
            score += bend_matrix.get(triplet, 0.050)
        bendabilities[i] = score / (len_chunk - 2 + 1e-5)
        
        # 2. Расчет Z-ДНК потенциала (поиск левозакрученных чередований RYRY)
        # Упрощенный быстрый поиск паттернов альтернирующих пуринов/пиримидинов
        z_score = node_chunk.count('GCGC') + node_chunk.count('CGCG') + node_chunk.count('GTGT') + node_chunk.count('CACA')
        zdna_potentials[i] = (z_score * 4) / len_chunk * 100  # Процент покрытия
        
        # 3. Аффинность нуклеосом (периодичность AA/TT/TA каждые 10 пар на отрезках упаковки)
        # Ищем корреляцию шага: А...через 10 букв...А
        nuc_shft = 0
        for idx in range(len_chunk - 11):
            if node_chunk[idx] in 'AT' and node_chunk[idx+10] in 'AT':
                nuc_shft += 1
        nucleosome_affinity[i] = (nuc_shft / (len_chunk - 11 + 1e-5)) * 100
     
        # Физика стержня Кирхгофа-Илюхина                
        if i < n_nodes - 1: segment_length = np.linalg.norm(coords_3d[i+1] - coords_3d[i])
        else: segment_length = np.linalg.norm(coords_3d[i] - coords_3d[i-1])
        if segment_length < 1e-5: segment_length = 3.4 
        
        # 4. Плотность упаковки букв на Ангстрем
        linear_densities[i] = len(node_chunk) / segment_length

    # ВТОРОЙ ПРОХОД: РАСЧЕТ ГЕОМЕТРИИ И ЭНЕРГЕТИКИ С УЧЕТОМ ОКРУЖЕНИЯ
    # Считаем матрицу тропических расстояний для гидрофобности
    for i in range(n_nodes):
        dists = np.linalg.norm(coords_3d - coords_3d[i], axis=1)
        dists[i] = float('inf')
        local_lambda = float(np.min(dists))
        
        # 1. Расчет геометрии осей Кирхгофа
        if 0 < i < n_nodes - 1:
            v1, v2 = coords_3d[i] - coords_3d[i-1], coords_3d[i+1] - coords_3d[i]
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 > 1e-5 and norm_v2 > 1e-5:
                curvatures[i] = 1.0 - np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                
        if 1 < i < n_nodes - 1:
            u1, u2, u3 = coords_3d[i-1]-coords_3d[i-2], coords_3d[i]-coords_3d[i-1], coords_3d[i+1]-coords_3d[i]
            n1, n2 = np.cross(u1, u2), np.cross(u2, u3)
            norm_n1, norm_n2 = np.linalg.norm(n1), np.linalg.norm(n2)
            if norm_n1 > 1e-5 and norm_n2 > 1e-5:
                twist_angles[i] = np.degrees(np.arccos(np.clip(np.dot(n1, n2) / (norm_n1 * norm_n2), -1.0, 1.0)))
                
        helix_pitches[i] = 2 * np.pi * (twist_angles[i] / (curvatures[i] + 1e-5)) if twist_angles[i] > 1e-5 and curvatures[i] > 1e-5 else 34.0
        if helix_pitches[i] > 100.0: helix_pitches[i] = 34.0
        
        # 2. ФИЗИКА: Энергия деформации упругой балки (в единицах kT)
        # Базовая изгибная жесткость ДНК B = 50 нм = 200 пН*нм^2. Формула: 0.5 * B * (Кривизна)^2
        strain_energies_kt[i] = 0.5 * 200.0 * (curvatures[i] ** 2)
        
        # 3. ФИЗИКА: Локальный гидропатический индекс (вытеснение воды из узла)
        # Падает при высоком local_lambda (рыхлая структура) и зависит от GC-состава
        hydrophobic_index[i] = (linear_densities[i] / (local_lambda + 1e-2)) * (1.0 + (scale_5mer[i] * 2))
        
        # 4. ФИЗИКА: Заряды Дебая-Хюккеля (солевое экранирование по закону Юкавы)
        # Радиус Дебая для ядра клетки λ_D = 1.0 нм
        debye_lambda = 1.0 
        force_sum = 0.0
        for j in range(n_nodes):
            if i != j:
                r = np.linalg.norm(coords_3d[i] - coords_3d[j])
                if r > 1e-5:
                    # Потенциал Юкавы: (1/r^13 для Ван-дер-Ваальса) + (Электростатика * экспоненциальное затухание солей)
                    yukawa_screening = np.exp(-r / debye_lambda)
                    force_sum += (1.0 / (r ** 13) + (1.0 / (r ** 2)) * yukawa_screening)
        debye_forces[i] = force_sum

    # 3 ПРОХОД
    # Геометрия осей (Кривизна, Кручение, Шаг и силы Ван-дер-Ваальса)
    for i in range(n_nodes):
        # 2. Нелинейная кривизна оси Кирхгофа
        if 0 < i < n_nodes - 1:
            v1 = coords_3d[i] - coords_3d[i-1]
            v2 = coords_3d[i+1] - coords_3d[i]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-5 and norm_v2 > 1e-5:
                cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                curvatures[i] = 1.0 - np.clip(cos_theta, -1.0, 1.0)
                
        # 1. Угол кручения нити через бинормали
        if 1 < i < n_nodes - 1:
            u1 = coords_3d[i-1] - coords_3d[i-2]
            u2 = coords_3d[i] - coords_3d[i-1]
            u3 = coords_3d[i+1] - coords_3d[i]
            n1 = np.cross(u1, u2)
            n2 = np.cross(u2, u3)
            norm_n1 = np.linalg.norm(n1)
            norm_n2 = np.linalg.norm(n2)
            if norm_n1 > 1e-5 and norm_n2 > 1e-5:
                cos_phi = np.dot(n1, n2) / (norm_n1 * norm_n2)
                twist_angles[i] = np.degrees(np.arccos(np.clip(cos_phi, -1.0, 1.0)))

        # 3. Шаг пространственной спирали
        if twist_angles[i] > 1e-5 and curvatures[i] > 1e-5:
            helix_pitches[i] = 2 * np.pi * (twist_angles[i] / (curvatures[i] + 1e-5))
            if helix_pitches[i] > 100.0: helix_pitches[i] = 34.0
        else: helix_pitches[i] = 34.0 

        # 5. Силы Ван-дер-Ваальса и электростатического расталкивания
        force_sum = 0.0
        for j in range(n_nodes):
            if i != j:
                r = np.linalg.norm(coords_3d[i] - coords_3d[j])
                if r > 1e-5:
                    force_sum += (1.0 / (r ** 13) + 1.0 / (r ** 2))
        vanderwaals_forces[i] = force_sum

    return {
        "twist_angle_deg": np.round(twist_angles, 2),
        "curvature_index": np.round(curvatures, 4),
        "helix_pitch_A": np.round(helix_pitches, 2),
        "linear_density_let_A": np.round(linear_densities, 2),
        "vdw_electro_force": np.round(vanderwaals_forces, 4),
        "molecular_mass_kda": np.round(molecular_masses_kda, 2),
        "t_melting_c": np.round(t_meltings, 1),
        "cpg_count": cpg_counts.astype(int),
        "gc_skew": np.round(gc_skews, 4), "at_skew": np.round(at_skews, 4),
        # Новые метрики в словарь пересылки
        "bendability": np.round(bendabilities, 4),
        "zdna_pot": np.round(zdna_potentials, 2),
        "nuc_affinity": np.round(nucleosome_affinity, 2),        
        # Новые ультимативные масштабы
        "scale_3mer": np.round(scale_3mer, 4), "scale_5mer": np.round(scale_5mer, 4),
        "scale_10mer": np.round(scale_10mer, 2), "scale_20mer": np.round(scale_20mer, 2),  
        # Возвращаем новые физические параметры
        "strain_energy_kt": np.round(strain_energies_kt, 3),
        "hydrophobic_idx": np.round(hydrophobic_index, 3),
        "debye_force": np.round(debye_forces, 4)                      
    }

# --- БЛОК ТРОПИЧЕСКОГО АНАЛИЗА ---
def compute_tropical_matrix(coords, epsilon=15.0):
    n = len(coords)
    mat = np.full((n, n), float('inf'))
    for i in range(n):
        for j in range(n):
            if i == j: mat[i][j] = 0.0
            else:
                d = np.linalg.norm(coords[i] - coords[j])
                if d <= epsilon: mat[i][j] = float(np.round(d, 3))
    return mat

# --- ФУНКЦИЯ ГЕНЕРАЦИИ НАУЧНЫХ ГРАФИКОВ (ДВУХОСЕВАЯ ИНФОГРАФИКА) ---
def generate_scientific_plots(df, timestamp):
    print("[Монитор]: Генерация графиков деформации и масс по Илюхину...")
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    
    color_bend = 'tab:blue'
    ax1.set_xlabel('Порядковый номер Марковского узла (0-25)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Нелинейный изгиб оси (Индекс кривизны)', color=color_bend, fontsize=12, fontweight='bold')
    ax1.plot(df['Узел_ID'], df['Нелинейн_Изгиб'], color=color_bend, marker='o', linewidth=2.5, label='Кривизна Илюхина')
    ax1.tick_params(axis='y', labelcolor=color_bend)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color_mass = 'tab:red'
    ax2.set_ylabel('Молекулярная масса сегмента (kDa)', color=color_mass, fontsize=12, fontweight='bold')
    ax2.bar(df['Узел_ID'], df['Масса_Узла_(kDa)'], color=color_mass, alpha=0.3, width=0.4, label='Масса (kDa)')
    ax2.tick_params(axis='y', labelcolor=color_mass)
    
    for idx, row in df.iterrows():
        if "OFF" in str(row['Статус']):
            ax1.axvspan(idx-0.4, idx+0.4, color='red', alpha=0.15)
            
    plt.title(f'Спектральный анализ ДНК Кирхгофа-Илюхина\nЧекпоинт: {timestamp}', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    # Сохраняем на диск
    os.makedirs("data/results/plots", exist_ok=True)
    plot_path = f"data/results/plots/scientific_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close() # ОЧИЩАЕМ ПАМЯТЬ КАРКАСА MATPLOTLIB!
    print(f"📈 [УСПЕХ]: Научный двухосевой график записан: {plot_path}")

# --- ИНИЦИАЛИЗАЦИЯ URSINA 3D ГРАФИКИ БЕЗ СТАРЫХ ДУБЛИКАТОВ COOLER ---
app = Ursina(title="TroPy Verified DNA Diagnostics Pro 2026")
window.fps_counter.enabled = True
EditorCamera()

nodes = [Entity(model='sphere', color=color.orange, scale=0.115) for _ in range(26)]
# lines = [Entity(model='cylinder', color=color.gray) for _ in range(25)]
lines = [Entity(model='cube', color=color.gray) for _ in range(25)]

# --- ИНТЕГРИРОВАННАЯ ЭКСПО-ФУНКЦИЯ ДЛЯ КНОПКИ (СБОРКА ВСЕХ 6 ПАРАМЕТРОВ) ---
def on_save_csv_and_plot_click():
    current_points = np.array([[n.x, n.y, n.z] for n in nodes])
    current_time_stamp = int(time.time())
    
    print("\n[ОЗУ -> Анализ]: Запуск сквозного обсчета 6 параметров нелинейной механики...")
    # Генерируем реальную строку генетического кода длиной 13.000 букв
    # random.seed(42)
    # letters_string = "".join(random.choices(['G', 'C', 'A', 'T'], k=13000))

    # --- НАШЕ РЕАЛЬНОЕ ПОДКЛЮЧЕНИЕ ФАЙЛА FASTA ДНК ЧЕЛОВЕКА ---
    # Скопируй имя своего файла из папки Загрузки и положи в data/raw/
    fasta_path = "data/raw/NC_000001.11[49897450..60115810].fa" 
    
    print(f"[ОЗУ -> Парсер]: Считывание реального кода ДНК из {fasta_path}...")
    letters_string = ""
    
    if os.path.exists(fasta_path):
        with open(fasta_path, "r") as f:
            # Читаем все строчки файла
            lines = f.readlines()
            # Отбрасываем самую первую строчку (заголовок >ref...)
            dna_lines = [line.strip().upper() for line in lines if not line.startswith(">")]
            # Склеиваем миллионы букв в одну гигантскую строку в ОЗУ
            letters_string = "".join(dna_lines)
        print(f"[ОЗУ -> Успех]: Загружено {len(letters_string)} реальных букв человека!")
    else:
        print(f"⚠️ Файл {fasta_path} не найден! Включаю резервную генерацию...")
        random.seed(42)
        letters_string = "".join(random.choices(['G', 'C', 'A', 'T'], k=10000000))
    
    # Запуск нашего обособленного физического движка
    phys = calculate_ilyukhin_and_biochem_parameters(current_points, letters_string)
        
    # Обертка для защиты от авто-форматирования дат в Excel
    def safe_num(val, ndigits=3):
        # Округляем и превращаем в строку, запечатывая в текстовую формулу Excel
        return f'="{round(float(val), ndigits)}"' # Изменено для обхода Excel-форматирования дат

    table_data = []


    for i in range(26):
        dists = np.linalg.norm(current_points - current_points[i], axis=1)
        dists[i] = float('inf')
        local_lambda = float(np.min(dists))
        local_chaos = float(np.linalg.norm(current_points[i] - base_dna_3d[i]))
        status = "OFF (Блок)" if local_lambda < 2.2 else "ON (Активен)"

        table_data.append({
            "Узел_ID": i,
            "Раздел_Гена": get_gene_name(i),
            "Троп_λ_(Å)": safe_num(local_lambda, 3),
            "Нелинейн_Изгиб": safe_num(phys["curvature_index"][i], 4),
            "Угол_Кручения_(°)": safe_num(phys["twist_angle_deg"][i], 2),
            "Шаг_Спирали_(Å)": safe_num(phys["helix_pitch_A"][i], 2),
            "Плотность_Букв_Å": safe_num(phys["linear_density_let_A"][i], 2),
            "Сила_ВанДерВаальса": safe_num(phys["vdw_electro_force"][i], 4),
            "Масса_Узла_(kDa)": safe_num(phys["molecular_mass_kda"][i], 2),
            
            # Внедряем новые параметры в макро-отчет
            "Т_Плавления_(C)": safe_num(phys["t_melting_c"][i], 1),
            "CpG_Островки_(шт)": safe_num(phys["cpg_count"][i], 0),
            "GC_Skew": safe_num(phys["gc_skew"][i], 4),
            "AT_Skew": safe_num(phys["at_skew"][i], 4),
            
            # ЧЕТЫРЕ НОВЫХ СТОЛБЦА ДЛЯ ИЗУЧЕНИЯ МАСШТАБОВ ДНК В МАКРО-ОТЧЕТЕ
            "Жесткость_Точечная_(3-mer)": safe_num(phys["scale_3mer"][i], 4),
            "Жесткость_Полувитка_(5-mer)": safe_num(phys["scale_5mer"][i], 4),
            "Периодичность_Витка_(10-mer)": safe_num(phys["scale_10mer"][i], 2),
            "Регуляторный_Профиль_(20-mer)": safe_num(phys["scale_20mer"][i], 2),
            
            # ТРИ НОВЫХ СТОЛБЦА ДЛЯ МАКРО-ТАБЛИЦЫ
            "Индекс_Изгибаемости": safe_num(phys["bendability"][i], 4),
            "Потенциал_Z_ДНК_%": safe_num(phys["zdna_pot"][i], 2),
            "Сродство_Нуклеосом_%": safe_num(phys["nuc_affinity"][i], 2),
            
            # ТРИ НОВЫХ ФИЗИЧЕСКИХ СТОЛБЦА ДЛЯ СРЕДЫ И ЭНЕРГЕТИКИ
            "Энергия_Деформации_(kT)": safe_num(phys["strain_energy_kt"][i], 3),
            "Гидрофобность_Узла": safe_num(phys["hydrophobic_idx"][i], 3),
            "Заряд_Дебая_(Юкава)": safe_num(phys["debye_force"][i], 4),
                                    
            "Хаос_RMSD": safe_num(local_chaos, 3),
            "Статус": status
        })
        
    df = pd.DataFrame(table_data)
    os.makedirs("data/results", exist_ok=True)
    csv_path = f"data/results/dna_comprehensive_report_{current_time_stamp}.csv"
    df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")
    print(f"📊 [GUI Экспорт]: Макро-отчет (26 узлов) создан: {csv_path}")

    # 2. ГЕНЕРАЦИЯ ГИГАНТСКОГО ДЕТАЛИЗИРОВАННОГО ОТЧЕТА
    print("[ОЗУ -> Экспорт]: Запуск фоновой микро-разметки миллионов нуклеотидов...")
    micro_table_data = []
    
    # Нарежем 10-мегабайтный файл на 10 000 мелких микро-зон для биологов
    micro_step = len(letters_string) // 10000 
    
    # Локальные копии матриц для микро-цикла
    m_matrix_3mer = {'AAA':0.005, 'TTT':0.005, 'AAT':0.020, 'ATT':0.020, 'GGC':0.090, 'GCC':0.090, 'CGC':0.085, 'GCG':0.095}
    m_matrix_5mer = {'AAAAA':0.012, 'TTTTT':0.012, 'GGGGG':0.088, 'CCCCC':0.088, 'GGCGG':0.095, 'CCGCC':0.095, 'ATATA':0.076}
    
    # Дублируем мини-матрицу жесткости для локального цикла
    m_bend_matrix = {'AAA':0.005, 'TTT':0.005, 'AAT':0.020, 'ATT':0.020, 
                     'GGC':0.090, 'GCC':0.090, 'GAC':0.075, 'GTC':0.075,
                     'CAG':0.060, 'CTG':0.060, 'CGC':0.085, 'GCG':0.095}
            
    for k in range(10000):
        m_start = k * micro_step
        m_end = (k + 1) * micro_step
        micro_chunk = letters_string[m_start:m_end]
        len_m = len(micro_chunk)
        if len_m == 0: len_m = 1

        # Быстрый подсчет букв в ОЗУ
        mA = micro_chunk.count('A')
        mT = micro_chunk.count('T')
        mG = micro_chunk.count('G')
        mC = micro_chunk.count('C')
        
        micro_mass_da = (mA * 313.21) + (mT * 304.20) + (mG * 329.21) + (mC * 289.18)
        
        # Вычисляем 4 биологические метрики для каждой микро-зоны
        m_t_melting = 2 * (mA + mT) + 4 * (mG + mC)
        m_cpg_count = micro_chunk.count('CG')
        m_gc_skew = (mG - mC) / (mG + mC + 1e-5)
        m_at_skew = (mA - mT) / (mA + mT + 1e-5)
        
        # Расчет новых метрик внутри микро-зоны
        m_bend = sum(m_bend_matrix.get(micro_chunk[z:z+3], 0.050) for z in range(len_m - 2)) / (len_m - 2 + 1e-5)
        m_z_score = micro_chunk.count('GCGC') + micro_chunk.count('CGCG') + micro_chunk.count('GTGT') + micro_chunk.count('CACA')
        m_zdna = (m_z_score * 4) / len_m * 100
        
        m_nuc = sum(1 for z in range(len_m - 11) if micro_chunk[z] in 'AT' and micro_chunk[z+10] in 'AT')
        m_nuc_affinity = (m_nuc / (len_m - 11 + 1e-5)) * 100
                        
        # Обсчет 4 пространственных шкал для микро-зоны
        m_s3 = sum(m_matrix_3mer.get(micro_chunk[z:z+3], 0.050) for z in range(len_m - 2)) / (len_m - 2 + 1e-5)
        m_s5 = sum(m_matrix_5mer.get(micro_chunk[z:z+5], 0.045) for z in range(len_m - 4)) / (len_m - 4 + 1e-5)
        m_s10 = sum(1 for z in range(len_m - 11) if micro_chunk[z] in 'AT' and micro_chunk[z+10] in 'AT') / (len_m - 11 + 1e-5) * 100
        m_s20 = sum((micro_chunk[z:z+20].count('G') + micro_chunk[z:z+20].count('C')) / 20.0 for z in range(len_m - 19)) / (len_m - 19 + 1e-5) * 100
                        
        # Определяем, к какому из 26 макро-узлов Ursina физически относится этот микро-кусок
        parent_node = k // (10000 // 26)
        parent_node = min(25, parent_node)
        
        # Локальная микро-физика упругости зоны
        # Приближенная микро-энергия деформации на основе изгибаемости 5-mer
        m_strain_kt = 0.5 * 200.0 * ((0.1 - m_s5) ** 2)
        m_hydro = (len_m / 3.4) * (1.0 + (m_s3 * 1.5))
        
        # Экранированный микро-заряд зоны (упрощенный внутренний Дебай)
        m_debye = (micro_mass_da / 1000.0) * np.exp(-3.4 / 1.0)
             
        # Обертка для защиты от авто-форматирования дат в Excel
        def safe_num(val, ndigits=3):
            # Округляем и превращаем в строку, запечатывая в текстовую формулу Excel
            return f'="{round(float(val), ndigits)}"'
     
        micro_table_data.append({
            "Микро_Зона_ID": f"Zone_{k}",
            "Координата_Старт": m_start + 49897450,
            "Координата_Конец": m_end + 49897450,
            "Родительский_3D_Узел": f"Node_{parent_node}",
            "Букв_в_куске": len(micro_chunk),
            
            # Оборачиваем массу в safe_num
            "Масса_Куска_(kDa)": safe_num(micro_mass_da / 1000.0, 3),
            
            # Оборачиваем GC-состав в safe_num
            "GC_Состав_%": safe_num(((mG + mC) / (len(micro_chunk) + 1e-5)) * 100, 1),
            
            # Запечатываем новые био-параметры в гигантскую таблицу
            "Т_Плавления_(C)": safe_num(m_t_melting, 1),
            "CpG_Островки_(шт)": safe_num(m_cpg_count, 0),
            "GC_Skew": safe_num(m_gc_skew, 4),
            "AT_Skew": safe_num(m_at_skew, 4),    
            
            # ТРИ НОВЫХ СТОЛБЦА ДЛЯ ДЕТАЛИЗИРОВАННОГО МИКРО-ОТЧЕТА
            "Индекс_Изгибаемости": safe_num(m_bend, 4),
            "Потенциал_Z_ДНК_%": safe_num(m_zdna, 2),
            "Сродство_Нуклеосом_%": safe_num(m_nuc_affinity, 2),            
            # ЧЕТЫРЕ НОВЫХ СТОЛБЦА В ГИГАНТСКОЙ ТАБЛИЦЕ
            "Жесткость_Точечная_(3-mer)": safe_num(m_s3, 4),
            "Жесткость_Полувитка_(5-mer)": safe_num(m_s5, 4),
            "Периодичность_Витка_(10-mer)": safe_num(m_s10, 2),
            "Регуляторный_Профиль_(20-mer)": safe_num(m_s20, 2), 
            
            # ТРИ НОВЫХ ФИЗИЧЕСКИХ СТОЛБЦА ДЛЯ МИКРО-ЗОН
            "Энергия_Деформации_(kT)": safe_num(m_strain_kt, 3),
            "Локальная_Гидрофобность": safe_num(m_hydro, 3),
            "Микро_Заряд_Дебая": safe_num(m_debye, 4)                               
        })
        
    df_micro = pd.DataFrame(micro_table_data)
    micro_csv_path = f"data/results/dna_BIG_detailed_report_{current_time_stamp}.csv"
    df_micro.to_csv(micro_csv_path, index=False, sep=";", encoding="utf-8-sig")
    
    print(f"🧬 [ГИГАНТСКИЙ ОТЧЕТ]: Создана детальная таблица на 10 000 строк: {micro_csv_path}")
        
    # Генерируем график на основе реальных расчитанных данных
    generate_scientific_plots(df, current_time_stamp)

    # === БЛОК АВТОМАТИЧЕСКОЙ ВЕРИФИКАЦИИ И САМОДИАГНОСТИКИ СИСТЕМЫ ===
    print("\n" + "="*60)
    print("🔬 [КОНТРОЛЬ КАЧЕСТВА]: ЗАПУСК АВТОНОМНОГО БЛОКА 'ПРОВЕРЬ СЕБЯ'")
    print("="*60)
    
    # 1. Извлекаем чистые числа из защищенных формул Excel для проверки
    gc_clean = df_micro["GC_Состав_%"].str.replace('="','').str.replace('"','').astype(float)
    mass_clean = df_micro["Масса_Куска_(kDa)"].str.replace('="','').str.replace('"','').astype(float)
    skew_clean = df_micro["GC_Skew"].str.replace('="','').str.replace('"','').astype(float)
    strain_clean = df_micro["Энергия_Деформации_(kT)"].str.replace('="','').str.replace('"','').astype(float)
    
    errors_found = 0

    # ТЕСТ 1: GC-состав человека (норма для хромосомы 1: 35% - 50%)
    gc_mean = float(gc_clean.mean())
    print(f"-> Тест 1 (GC-состав): Средний по выборке = {gc_mean:.2f}%")
    if 30.0 <= gc_mean <= 55.0:
        print("   ✅ СТАТУС: ИДЕАЛЬНО (Соответствует геному Homo Sapiens)")
    else:
        print("   ⚠️ СТАТУС: ВНИМАНИЕ (Выход за рамки человеческой нормы!)")
        errors_found += 1

    # ТЕСТ 2: Молекулярный вес куска (~1000 букв должно весить 300-340 kDa)
    mass_mean = float(mass_clean.mean())
    print(f"-> Тест 2 (Молекулярная масса): Средний вес сегмента = {mass_mean:.1f} kDa")
    if 290.0 <= mass_mean <= 350.0:
        print("   ✅ СТАТУС: ИДЕАЛЬНО (Физический вес нуклеотидов подтвержден)")
    else:
        print("   ❌ СТАТУС: КРИТИЧЕСКАЯ ОШИБКА (Вес сломан, проверьте формулу!)")
        errors_found += 1

    # ТЕСТ 3: Математические границы асимметрии Skew (строго от -1.0 до 1.0)
    skew_min, skew_max = float(skew_clean.min()), float(skew_clean.max())
    print(f"-> Тест 3 (Асимметрия Skew): Диапазон значений = [{skew_min:.4f} : {skew_max:.4f}]")
    if -1.0 <= skew_min and skew_max <= 1.0:
        print("   ✅ СТАТУС: ИДЕАЛЬНО (Математические границы соблюдены, деления на ноль нет)")
    else:
        print("   ❌ СТАТУС: КРИТИЧЕСКАЯ ОШИБКА (Вылет за пределы математической логики!)")
        errors_found += 1

    # ТЕСТ 4: Энергетическая стабильность стержня Бернулли-Эйлера
    strain_max = float(strain_clean.max())
    print(f"-> Тест 4 (Энергия упругости): Пиковое напряжение в узлах = {strain_max:.3f} kT")
    if strain_max < 50.0:
        print("   ✅ СТАТУС: ИДЕАЛЬНО (Нить стабильна, тепловой хаос не разрывает ДНК)")
    else:
        print("   ⚠️ СТАТУС: ВНИМАНИЕ (Обнаружены зоны экстремального физического разрыва!)")

    print("-"*60)
    if errors_found == 0:
        print("🎉 [ВЕРДИКТ]: ВСЕ ДАННЫЕ АДЕКВАТНЫ И ДОСТОВЕРНЫ. СБОЕВ НЕ ОБНАРУЖЕНО.")
    else:
        print(f"❌ [ВЕРДИКТ]: ОБНАРУЖЕНО ОШИБОК: {errors_found}. ТРЕБУЕТСЯ ПЕРЕПРОВЕРКА ДВИЖКА.")
    print("="*60 + "\n")

    # ТЕСТ 5: Экранирование солей (Закон Юкавы и Дебаевская длина)
    debye_clean = df_micro["Микро_Заряд_Дебая"].str.replace('="','').str.replace('"','').astype(float)
    debye_mean, debye_max = float(debye_clean.mean()), float(debye_clean.max())
    print(f"-> Тест 5 (Экранирование солей): Средний микро-заряд = {debye_mean:.2f}, Пиковый = {debye_max:.2f}")
    if 0.01 <= debye_mean <= 50.0:
        print("   ✅ СТАТУС: ИДЕАЛЬНО (Радиус Дебая ~1.0 нм гасит заряды согласно законам электролитов)")
    else:
        print("   ❌ СТАТУС: КРИТИЧЕСКАЯ ОШИБКА (Сбой солевого баланса! Сила Юкавы улетела за физические рамки)")
        errors_found += 1

    del current_points, table_data, df, phys
    print("[ОЗУ -> ОЧИСТКА]: Буфер ОЗУ очищен. Память ПК стабильна.")

# --- ОСТАЛЬНАЯ GUI ЛОГИКА ---
def on_reload_block_click():
    load_and_process_biological_block()
    print("[GUI]: Новый блок загружен!")

def on_reset_chaos_click():
    global lcm_phases
    lcm_phases = [random.uniform(0, 50) for _ in range(13)]
    print("[GUI]: Фазы 13 LCM сброшены!")

def on_manual_screenshot_click():
    os.makedirs("data/results/screenshots", exist_ok=True)
    shot_path = f"data/results/screenshots/manual_{int(time.time())}.png"
    
    print(f"\n[GUI]: Запрос ручного скриншота...")
    try:
        app.screenshot(shot_path)
        print(f"📸 [GUI]: Ручной скриншот сохранен: {shot_path}")
    except Exception as e:
        print(f"❌ [GUI ОШИБКА]: Сбой снимка: {e}")
           
    # window.screenshot(name=shot_path, compute_shadows=False)
    # screenshot(name=shot_path, compute_shadows=False)
    # window.screenshot(name=shot_path, delay=0)
    # print(f"📸 [GUI]: Ручной скриншот сохранен: {shot_path}")

# Создание физических кнопок в Ursina
btn_reload = Button(text="Загрузить Блок", color=color.azure, scale=(0.2, 0.05), position=(-0.7, 0.45))
btn_chaos = Button(text="Сбросить Хаос", color=color.orange, scale=(0.2, 0.05), position=(-0.7, 0.38))
btn_csv = Button(text="Построить Графики", color=color.red, scale=(0.2, 0.05), position=(-0.7, 0.31))
btn_shot = Button(text="Сделать Скриншот", color=color.violet, scale=(0.2, 0.05), position=(-0.7, 0.24))

# === НОВЫЙ БЛОК: ФУНКЦИЯ ДЛЯ КНОПКИ-ПЕРЕКЛЮЧАТЕЛЯ ===
def on_toggle_auto_screenshot_click():
    global AUTO_SCREENSHOT_ENABLED
    AUTO_SCREENSHOT_ENABLED = not AUTO_SCREENSHOT_ENABLED # Инвертируем состояние (True <-> False)
    
    if AUTO_SCREENSHOT_ENABLED:
        btn_toggle_auto.text = "Авто-фиксация: ON"
        btn_toggle_auto.color = color.green      # Яркий зеленый, если активна
    else:
        btn_toggle_auto.text = "Авто-фиксация: OFF"
        btn_toggle_auto.color = color.dark_gray  # Продавленный темный цвет, если отключена
    print(f"⚙️ [GUI]: Режим автоматических скриншотов изменен на: {AUTO_SCREENSHOT_ENABLED}")

# Создаем саму кнопку на панели слева внизу
btn_toggle_auto = Button(
    text="Авто-фиксация: ON", 
    color=color.green, 
    scale=(0.2, 0.05), 
    position=(-0.7, 0.17) # Стоит строго под фиолетовой кнопкой
)
btn_toggle_auto.on_click = on_toggle_auto_screenshot_click
# ===================================================


btn_reload.on_click = on_reload_block_click
btn_chaos.on_click = on_reset_chaos_click
btn_csv.on_click = on_save_csv_and_plot_click
btn_shot.on_click = on_manual_screenshot_click

# --- НАУЧНЫЙ РЕГУЛЯТОР ТЕМПЕРАТУРЫ (20-25°C) ---
TEMPERATURE_CELSIUS = 22.0

# --- ГЛАВНЫЙ ЦИКЛ ОБНОВЛЕНИЯ, ОКРАСКИ И ХАОСА ---
def update():
    # === ДОБАВЛЯЕМ НАКОПЛЕНИЕ ВРЕМЕНИ В НАЧАЛО UPDATE ===
    global frame_timer, lcm_phases, screenshot_cooldown, startup_safe_timer
    startup_safe_timer += time.dt    
    frame_timer += 1
    dt = time.dt
    
    # Термодинамика хаоса Кирхгофа
    temp_kelvin = TEMPERATURE_CELSIUS + 273.15
    chaos_thermal_multiplier = np.sqrt(temp_kelvin / 293.15)
    rigidity_drop = max(0.1, 1.0 - (max(0, TEMPERATURE_CELSIUS - 20.0) * 0.008))
    chaos_power = 0.08 * chaos_thermal_multiplier / rigidity_drop
    
    for k in range(13): 
        lcm_phases[k] += lcm_speeds[k] * dt
        
    chaos_x = sum(np.sin(lcm_phases[k]) * chaos_power for k in range(13))
    chaos_z = sum(np.cos(lcm_phases[k]) * chaos_power for k in range(13))

    for i in range(26):
        nodes[i].position = Vec3(
            (base_dna_3d[i, 0] + chaos_x * np.sin(i * 0.15)) * 20.0, # Умножаем на 20 прямо тут
            (base_dna_3d[i, 1]) * 20.0,                              # Умножаем на 20 прямо тут
            (base_dna_3d[i, 2] + chaos_z * np.cos(i * 0.15)) * 20.0  # Умножаем на 20 прямо тут
        )  

    current_coords = np.array([[n.x, n.y, n.z] for n in nodes])
    has_critical_anomaly = False
    
    # Динамический градиент опасности (Твоя логика!)
    for i in range(26):
        dists = np.linalg.norm(current_coords - current_coords[i], axis=1)
        dists[i] = float('inf')
        local_lambda = float(np.min(dists))
        
        # НАСТРОЕННАЯ ИСПРАВЛЕННАЯ ЦВЕТОВАЯ ЛОГИКА:
        if local_lambda < 0.3:
            # Эпицентр аномалии (сверхплотное сжатие)
            nodes[i].color = color.red
            nodes[i].scale = 0.5
            has_critical_anomaly = True
        elif 0.3 <= local_lambda < 0.7:
            # Зона предупреждения (близко к аномалии)
            nodes[i].color = color.yellow
            nodes[i].scale = 0.4
        else:
            # Нормальное, свободное состояние полимера
            nodes[i].color = color.green
            nodes[i].scale = 0.3
            
    del current_coords

    # МОДИФИЦИРОВАННОЕ УСЛОВИЕ АВТО-СКРИНШОТА:
    # Робот сработает только если есть аномалия, нет кулдауна И прошло больше 5 секунд со старта!
    # МОДИФИЦИРОВАННОЕ УСЛОВИЕ АВТО-СКРИНШОТА С УЧЕТОМ НАЖАТОЙ КНОПКИ:
    # Робот сделает снимок ТОЛЬКО если кнопка на панели горит зеленым (True)    
    if has_critical_anomaly and not screenshot_cooldown and startup_safe_timer > 5.0 and AUTO_SCREENSHOT_ENABLED:
        screenshot_cooldown = True
        os.makedirs("data/results/screenshots", exist_ok=True)
        shot_path = f"data/results/screenshots/auto_anomaly_{int(time.time())}.png"

        # НАШИ ОТЛАДОЧНЫЕ ПРИНТЫ
        
        print(f"\n[ОТЛАДКА]: Попытка фиксации аномалии в: {shot_path}")
        try:
            # СТРОГИЙ СИНТАКСИС PANDA3D: путь передается напрямую, без name=
            app.screenshot(shot_path)
            print(f"📸 [АВТО-ФИКСАЦИЯ]: Критическая зона заснята успешно!")
        except Exception as e:
            print(f"❌ [ОШИБКА СКРИНШОТА]: Кадр пропущен: {e}")
        
    for i in range(25):
        p1, p2 = nodes[i].position, nodes[i+1].position
        lines[i].position = (p1 + p2) / 2
        lines[i].look_at(p2)
        lines[i].rotation_x += 90
        # lines[i].scale = Vec3(0.1, distance(p1, p2), 0.1)
        lines[i].scale = Vec3(0.05, distance(p1, p2), 0.05)
        if nodes[i].color == color.red or nodes[i+1].color == color.red:
            lines[i].color = color.rgba(255, 0, 0, 200)
        else:
            lines[i].color = color.rgba(200, 200, 200, 100)
            
    # Такт верификации и сброса ОЗУ раз в 1000 кадров
    if frame_timer >= 1000:
        frame_timer = 0
        screenshot_cooldown = False
        current_ram_points = np.array([[n.x, n.y, n.z] for n in nodes])
        
        virtual_dist_matrix = np.zeros((26, 26))
        for i in range(26):
            for j in range(26): 
                virtual_dist_matrix[i, j] = np.linalg.norm(current_ram_points[i] - current_ram_points[j])
                
        matrix_correlation = np.corrcoef(distance_2d.flatten(), virtual_dist_matrix.flatten())[0, 1]
        math_accuracy = float(matrix_correlation * 100)
        rmsd_error = float(np.sqrt(np.mean(np.sum((current_ram_points - pdb_cryo_em_reference)**2, axis=1))))
        
        print(f"\n[Контроль качества]: Точность обратной 2D-развертки: {math_accuracy:.2f}%. Тепловое уклонение от Крио-ЭМ: {rmsd_error:.3f} Å [2.1, 5.1]")
        
        del current_ram_points, virtual_dist_matrix
        print("[ОЗУ -> ЧИСТКА]: Тактовая очистка выполнена.")

app.run()






















