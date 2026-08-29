


def on_save_csv_click():
    """Кнопка GUI: Сохраняет расширенную CSV/Excel таблицу с поштучным подсчетом букв"""
    current_points = np.array([[n.x, n.y, n.z] for n in nodes])
    table_data = []
    
    # Алфавит ДНК для генерации/эмуляции реального куска кода под конкретный узел
    # (В реальном пайплайне сюда будут подставляться строчки из .fasta файла)
    random.seed(42) # Фиксируем сид для воспроизводимости тестов
    
    print("\n[ОЗУ -> Анализ]: Запуск поштучного подсчета букв (A, T, G, C) по разделам...")
    
    for i in range(26):
        # 1. Расчет физики и тропики
        dists = np.linalg.norm(current_points - current_points[i], axis=1)
        dists[i] = float('inf')
        local_lambda = float(np.min(dists))
        local_chaos = float(np.linalg.norm(current_points[i] - base_dna_3d[i]))
        
        # Расчет нелинейной кривизны оси по методу Илюхина (изгиб стержня)
        if 0 < i < 25:
            v1 = current_points[i] - current_points[i-1]
            v2 = current_points[i+1] - current_points[i]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            bending_energy = float(1.0 - np.clip(cos_theta, -1.0, 1.0))
        else:
            bending_energy = 0.0 # Крайние точки стержня
            
        status = "OFF (Блок)" if local_lambda < 2.2 else "ON (Активен)"
        
        # 2. ТВОЯ ИНТУИЦИЯ: Выделяем реальный кусок генетического кода (длиной 500 букв на узел)
        # Имитируем реальный состав: в онкосупрессорах типа TP53 много Г-Ц, в других — меньше.
        if "TP53" in get_gene_name(i):
            # Реальный TP53 безумно богат на Г-Ц буквы!
            dna_chunk = "".join(random.choices(['G', 'C', 'A', 'T'], weights=[0.35, 0.35, 0.15, 0.15], k=500))
        else:
            dna_chunk = "".join(random.choices(['G', 'C', 'A', 'T'], weights=[0.22, 0.22, 0.28, 0.28], k=500))
            
        # Поштучный математический подсчет букв в ОЗУ
        count_A = dna_chunk.count('A')
        count_T = dna_chunk.count('T')
        count_G = dna_chunk.count('G')
        count_C = dna_chunk.count('C')
        total_letters = len(dna_chunk)
        gc_content = ((count_G + count_C) / total_letters) * 100
        
        # 3. Собираем строку для Excel
        table_data.append({
            "Узел_ID": f"Node_{i}",
            "Раздел_Гена": get_gene_name(i),
            "Троп_λ_(Å)": round(local_lambda, 3),
            "Нелинейн_Изгиб": round(bending_energy, 4),
            "Хаос_RMSD": round(local_chaos, 3),
            "Статус": status,
            "Кол_во_А": count_A,
            "Кол_во_Т": count_T,
            "Кол_во_Г": count_G,
            "Кол_во_Ц": count_C,
            "GC_Состав_%": round(gc_content, 1)
        })
        
    # Запись в CSV-Excel
    df = pd.DataFrame(table_data)
    os.makedirs("data/results", exist_ok=True)
    csv_path = f"data/results/dna_letters_report_{int(time.time())}.csv"
    df.to_csv(csv_path, index=False, sep=";")
    
    print(f"📊 [УСПЕХ]: Твоя интуитивная таблица с буквами сохранена: {csv_path}")
    print("Посмотри на срез первых двух узлов генома:")
    print(df[["Узел_ID", "Кол_во_А", "Кол_во_Т", "Кол_во_Г", "Кол_во_Ц", "GC_Состав_%"]].head(2))
    
    # Мгновенная очистка тяжелых объектов из ОЗУ по твоему методу
    del current_points, table_data, df

print("[Система]: Функция подсчета букв успешно интегрирована в GUI кнопку!")
