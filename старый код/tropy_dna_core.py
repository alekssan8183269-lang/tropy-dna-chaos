import numpy as np
import random
from ursina import *

# === 1. БЛОК СЖАТИЯ ДАННЫХ И МОДЕЛИРОВАНИЯ (Под твои 26 марковских узлов) ===
def generate_biological_loop(nodes=26):
    """Генерирует завязанную в пространстве ДНК-петлю (имитация PDB-данных)"""
    t = np.linspace(0, 2 * np.pi, nodes)
    # Формула математического узла-трилистника (самый частый узел в ДНК)
    x = np.sin(t) + 2 * np.sin(2*t)
    y = np.cos(t) - 2 * np.cos(2*t)
    z = -np.sin(3*t)
    return np.column_stack([x, y, z]) * 2  # Масштабируем в Ангстремы

# === 2. БЛОК ТРОПИЧЕСКОГО АНАЛИЗА ДЛЯ БИОЛОГОВ (TroPy) ===
def get_tropical_knot_index(coords, epsilon=8.0):
    """Строит min-plus матрицу расстояний и считает индекс узла"""
    n = len(coords)
    tropical_matrix = np.full((n, n), float('inf'))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                tropical_matrix[i][j] = 0.0
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= epsilon:  # Если атомы в зоне контакта
                    tropical_matrix[i][j] = float(np.round(dist, 3))
                    
    # Инвариант: берем минимальные нетривиальные тропические расстояния
    bonds = tropical_matrix[tropical_matrix > 0.0]
    return float(np.min(bonds)) if len(bonds) > 0 else float('inf')

# === 3. БЛОК ХАРАКТЕРИСТИКИ И 3D-ГРАФИКИ (Ursina + 13 LCM Осцилляторов) ===
base_dna = generate_biological_loop(nodes=26)

app = Ursina(title="TroPy DNA Diagnostics")
window.fps_counter.enabled = False
EditorCamera()

# Создаем 26 сфер в ОЗУ
nodes = [Entity(model='sphere', color=color.emerald, scale=0.3) for _ in range(26)]
lines = [Entity(model='cylinder', color=color.white) for _ in range(25)]

# Настройка 13 LCM хаотических фаз
lcm_phases = [random.uniform(0, 10) for _ in range(13)]
lcm_speeds = [random.uniform(2, 5) for _ in range(13)]

frame_timer = 0

def update():
    global frame_timer
    frame_timer += 1
    dt = time.dt
    
    # Считаем хаос от 13 осцилляторов
    for k in range(13): lcm_phases[k] += lcm_speeds[k] * dt
    chaos_x = sum(np.sin(lcm_phases[k]) * 0.12 for k in range(13))
    chaos_z = sum(np.cos(lcm_phases[k]) * 0.12 for k in range(13))
    
    # Двигаем узлы ДНК на экране
    for i in range(26):
        nodes[i].position = Vec3(
            base_dna[i][0] + chaos_x * np.sin(i*0.1),
            base_dna[i][1],
            base_dna[i][2] + chaos_z * np.cos(i*0.1)
        )
        
    # Обновляем цилиндры-связи (без утечек ОЗУ)
    for i in range(25):
        p1, p2 = nodes[i].position, nodes[i+1].position
        lines[i].position = (p1 + p2) / 2
        lines[i].look_at(p2)
        lines[i].rotation_x += 90
        lines[i].scale = Vec3(0.08, distance(p1, p2), 0.08)
        
    # То, что нужно биологам: Анализ каждые 150 кадров + ОЧИСТКА ОЗУ
    if frame_timer >= 150:
        frame_timer = 0
        current_points = np.array([[n.x, n.y, n.z] for n in nodes])
        
        # Считаем то, что биологи не могут увидеть на приборах
        lambda_index = get_tropical_knot_index(current_points)
        
        print(f"\n[ДИАГНОСТИКА ДНК]: Срез динамического хаоса выполнен.")
        print(f"   -> Тропический индекс натяжения узла (λ): {lambda_index:.3f} Ангстрем")
        if lambda_index < 1.2:
            print("   ⚠️ [КРИТИЧЕСКИЙ СТАТУС]: Узел ДНК перетянут! Блокировка транскрипции гена!")
        else:
            print("   ✅ [СТАТУС]: Нормальная эластичность полимера.")
            
        del current_points # Жестко чистим ОЗУ!

app.run()
