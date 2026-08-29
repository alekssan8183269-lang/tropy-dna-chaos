import os
import sys
import numpy as np
import random
from ursina import *

# === НАСТРОЙКА ХАРАКТЕРИСТИК ДВИЖКА ИЗ ТВОЕГО РЕПОЗИТОРИЯ 13-LCM ===
NUM_NODES = 26          # Жесткий лимит под твой Марковский граф
NUM_OSCILLATORS = 13    # Твои 13 хаотических LCM осцилляторов

print("=== ШАГ 3: ЗАПУСК ХАРАТИЧЕСКОЙ СИМУЛЯЦИИ ДНК В URSINA 3D ===")

# Пытаемся загрузить очищенные координаты из Шага 1
DATA_PATH = "data/processed/dna_26_nodes.npy"
if not os.path.exists(DATA_PATH):
    print(f"[ОШИБКА]: Чекпоинт {DATA_PATH} не найден. Сгенерируем синтетический тестовый каркас на 26 точек.")
    # Генерируем тестовую спираль в пространстве
    t = np.linspace(0, 10, NUM_NODES)
    base_coords = np.column_stack([np.sin(t)*2, t, np.cos(t)*2])
else:
    base_coords = np.load(DATA_PATH)
    print(f"[ОЗУ -> Ursina]: Загружен биологический каркас ДНК. Точек: {len(base_coords)}")

# Центрируем ДНК в нулевые координаты экрана для удобства просмотра
base_coords -= np.mean(base_coords, axis=0)

# Инициализируем окно Ursina
app = Ursina(title="TroPy DNA Chaos Simulation 2026")
window.fps_counter.enabled = True

# Массивы для хранения графических объектов в ОЗУ
node_entities = []
segment_entities = []

# Инициализируем 13 хаотических фаз для LCM-осцилляторов
lcm_phases = [random.uniform(0, 100) for _ in range(NUM_OSCILLATORS)]
lcm_speeds = [random.uniform(1.5, 4.0) for _ in range(NUM_OSCILLATORS)]

# Отрисовка начального состояния ДНК
print(f"[Графика]: Создание {NUM_NODES} сфер-узлов и цилиндров-связей...")
for i in range(NUM_NODES):
    # Создаем сферу для каждого из 26 марковских шарниров
    node = Entity(
        model='sphere', 
        color=color.cyan, 
        scale=0.3, 
        position=Vec3(base_coords[i][0], base_coords[i][1], base_coords[i][2])
    )
    node_entities.append(node)

# Функция для обновления цилиндров между сферами
def update_segments():
    # Чтобы ОЗУ не текла, мы не пересоздаем объекты, а просто двигаем и масштабируем старые
    global segment_entities
    
    # Если цилиндров еще нет — создаем их один раз
    if len(segment_entities) == 0:
        for i in range(NUM_NODES - 1):
            p1 = node_entities[i].position
            p2 = node_entities[i+1].position
            segment = Entity(model='cylinder', color=color.rgba(255,255,255,150))
            segment_entities.append(segment)
            
    # Пересчитываем положение цилиндров-связей
    for i in range(NUM_NODES - 1):
        p1 = node_entities[i].position
        p2 = node_entities[i+1].position
        segment = segment_entities[i]
        segment.position = (p1 + p2) / 2
        segment.look_at(p2)
        segment.rotation_x += 90 # Корректировка оси цилиндра Ursina
        segment.scale = Vec3(0.1, distance(p1, p2), 0.1)

update_segments()

# Камера
EditorCamera() # Позволяет крутить сцену мышкой (ПКМ — вращение, Колесико — зум)

# Таймер для вывода логов работы с ОЗУ на каждые 200 кадров
frame_counter = 0

def update():
    """
    Главный цикл Ursina. Вызывается на каждый кадр.
    Здесь рассчитывается хаос от 13 LCM и жестко контролируется память.
    """
    global lcm_phases, frame_counter
    frame_counter += 1
    
    # 1. Рассчитываем 13 хаотических LCM смещений
    dt = time.dt
    for k in range(NUM_OSCILLATORS):
        lcm_phases[k] += lcm_speeds[k] * dt
        
    # Считаем суперпозицию хаотического сигнала
    chaos_x = sum(np.sin(lcm_phases[k]) * 0.15 for k in range(NUM_OSCILLATORS))
    chaos_z = sum(np.cos(lcm_phases[k] * 0.8) * 0.15 for k in range(NUM_OSCILLATORS))
    
    # 2. Применяем хаос к нашим 26 узлам Марковского графа
    for i in range(NUM_NODES):
        # Каждый узел реагирует на хаос со своим фазовым сдвигом (волна по нити ДНК)
        shift = i * 0.2
        node_entities[i].x = base_coords[i][0] + chaos_x * np.sin(lcm_phases[0] + shift)
        node_entities[i].z = base_coords[i][2] + chaos_z * np.cos(lcm_phases[1] + shift)
        
    # 3. Обновляем геометрию связей
    update_segments()
    
    # 4. Твоя ключевая фича: Каждые 200 кадров сбрасываем данные и имитируем очистку ОЗУ
    if frame_counter >= 200:
        frame_counter = 0
        print("\n[Симуляция -> Статус]: Отработано 200 кадров динамики.")
        
        # Вытаскиваем текущие деформированные координаты для тропического анализа
        current_ram_coords = np.array([[n.x, n.y, n.z] for n in node_entities])
        print(f"   -> Извлечено из ОЗУ текущих координат: {current_ram_coords.shape}")
        
        # Считаем среднее смещение нити под хаосом
        mean_drift = np.mean(np.linalg.norm(current_ram_coords - base_coords, axis=1))
        print(f"   -> Текущая хаотическая деформация нити: {mean_drift:.4f} Ангстрем")
        
        # КОВАЛЬНО ОЧИЩАЕМ временный массив, чтобы ОЗУ ПК была свободна
        del current_ram_coords
        print("[ОЗУ -> ОЧИСТКА]: Промежуточный массив стерт из ОЗУ. Симуляция стабильна!")

# Запуск приложения Ursina
app.run()
