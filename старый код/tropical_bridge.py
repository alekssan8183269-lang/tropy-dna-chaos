import os
import sys
import numpy as np

# Симулируем константу бесконечности для min-plus полукольца
TROPICAL_INF = float('inf')

def load_dna_coordinates(file_path="data/processed/dna_26_nodes.npy"):
    """Загружает сжатые 3D-координаты ДНК из чекпоинта"""
    if not os.path.exists(file_path):
        print(f"[КРИТИЧЕСКАЯ ОШИБКА]: Файл {file_path} не найден! Сначала запусти Шаг 1.")
        sys.exit(1)
    
    coords = np.load(file_path)
    print(f"[ОЗУ -> Загрузка]: Успешно загружен массив координат ДНК.")
    print(f"   -> Форма массива в памяти: {coords.shape} (26 узлов, 3 координаты)")
    return coords

def build_min_plus_distance_matrix(coords, epsilon=12.0):
    """
    Строит классическую матрицу евклидовых расстояний 
    и адаптирует её под правила min-plus алгебры TroPy.
    """
    n = len(coords)
    print(f"\n[Вычисления]: Инициализация тропической матрицы {n}x{n}...")
    
    # В min-plus алгебре нейтральный элемент по сложению — это бесконечность.
    # Заполняем матрицу бесконечностями, имитируя отсутствие связей.
    tropical_matrix = np.full((n, n), TROPICAL_INF)
    
    connections_count = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Расстояние от узла до самого себя в тропической геометрии равно 0
                tropical_matrix[i][j] = 0.0
            else:
                # Считаем реальное физическое расстояние между шарнирами ДНК (в Ангстремах)
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # Физический фильтр (Эпсилон-окрестность):
                # Если атомы слишком далеко, они не образуют петлю/узел. Режем связь.
                if dist <= epsilon:
                    tropical_matrix[i][j] = float(np.round(dist, 4))
                    connections_count += 1
                    
    print(f"[Вычисления]: Матрица построена.")
    print(f"   -> Всего возможных связей (физических контактов): {connections_count}")
    return tropical_matrix

def compute_tropical_eigenvalue_approximation(matrix):
    """
    Пример тропического инварианта для твоего TroPy.
    В min-plus алгебре аналог собственного значения матрицы графа — 
    это минимальный средний вес цикла (минимальная плотность петли).
    """
    n = matrix.shape[0]
    # Находим минимальные нетривиальные расстояния между соседними узлами
    valid_distances = matrix[matrix > 0.0]
    if len(valid_distances) == 0:
        return TROPICAL_INF
        
    # Базовая оценка «плотности» скручивания узла
    tropical_lambda = np.min(valid_distances)
    return float(np.round(tropical_lambda, 4))

def process_batch_and_clear_ram(pdb_id="1bna"):
    """
    Главный конвейер Фазы 2: Загружает данные, считает тропический каркас,
    выводит результат в консоль, сбрасывает на диск и КОВАЛЬНО ЧИСТИТ ОЗУ.
    """
    print(f"\n=== ЗАПУСК ТРОПИЧЕСКОГО АНАЛИЗА ДЛЯ МОЛЕКУЛЫ: {pdb_id.upper()} ===")
    
    # 1. Загрузка данных в ОЗУ
    dna_coords = load_dna_coordinates()
    
    # 2. Построение min-plus матрицы
    # ε=12 Ангстрем — стандартная зона топологического контакта для ДНК
    trop_matrix = build_min_plus_distance_matrix(dna_coords, epsilon=12.0)
    
    # Показываем срез матрицы для отладки твоего TroPy
    print("\n[Отладка TroPy]: Срез верхнего левого угла матрицы (5x5):")
    print(trop_matrix[:5, :5])
    
    # 3. Расчет тропического инварианта (цифрового отпечатка узла)
    trop_lambda = compute_tropical_eigenvalue_approximation(trop_matrix)
    print(f"\n[Результат TroPy]: Рассчитан тропический инвариант скручивания (λ) = {trop_lambda}")
    
    # 4. Сохраняем результат на жесткий диск как чекпоинт
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{pdb_id}_tropical_meta.txt")
    
    with open(result_path, "w") as f:
        f.write(f"PDB_ID: {pdb_id}\n")
        f.write(f"Tropical_Lambda: {trop_lambda}\n")
        f.write(f"Matrix_Shape: {trop_matrix.shape}\n")
        
    print(f"[Чекпоинт]: Результаты улетели на диск: {result_path}")
    
    # 5. КРИТИЧЕСКИЙ ШАГ: То, о чем ты говорил. Полная очистка ОЗУ от тяжелых матриц
    print("\n[ОЗУ -> ОЧИСТКА]: Стираю отработанную матрицу и координаты из оперативной памяти...")
    
    del dna_coords
    del trop_matrix
    
    print("[ОЗУ -> СТАТУС]: Память чиста! Компьютер готов к следующему шагу хаотической симуляции.")

if __name__ == "__main__":
    # Запускаем полный цикл вычислений
    process_batch_and_clear_ram(pdb_id="1bna")
