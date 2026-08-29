# TroPy DNA Chaos: 3D Genome Reconstruction & Nonlinear Dynamics Framework

## 🔬 Обзор проекта (Independent Research)
Эксперименты класса Hi-C/Micro-C предоставляют молекулярным биологам огромные массивы данных, но они ограничены плоскими 2D-картами контактов (слепая зона в 3D) [2.1]. 

Данный программный комплекс решает **обратную задачу конформации генома** [2.1]: он в реальном времени считывает «грязные» двухмерные научные матрицы (формат `.cool`), математически разворачивает их в пространственные 3D-координаты [2.2, 3.1] и моделирует их нелинейную динамику под воздействием внутриклеточного теплового хаоса при фиксированной температуре (20–25°C).

Проект объединяет три фундаментальные дисциплины:
1. **Тропическая геометрия (`min-plus` алгебра)** — для сверхбыстрого расчета матриц расстояний и поиска топологических инвариантов скручивания ДНК без тяжелых интегралов.
2. **Нейроморфные хаотические системы (13-LCM-Markov-Crossbar)** — симуляция броуновских флуктуаций ядра через суперпозицию 13 хаотических осцилляторов на 26-узловом Марковском графе.
3. **Геометрически нелинейная механика упругих стержней** — развитие идей профессора А. А. Илюхина (Таганрогская школа механики стержней Кирхгофа) в применении к молекуле ДНК.

## 🚀 Ключевые фичи и аналитические маркеры
* **Решение обратной задачи без заглушек:** Полный автоматический цикл развертки реальных биологических файлов `.cool` в 3D-модели (алгоритм MDS) [2.2, 3.1].
* **Химико-физический синтез:** Поштучный подсчет нуклеотидов (A, T, G, C) по разделам генома с автоматическим вычислением реальной молекулярной массы каждого узла в Килодальтонах (kDa) [4.1].
* **Жесткий контроль ОЗУ:** Оптимизированный буфер данных. Тяжелые проверочные матрицы автоматически выгружаются в текстовые чекпоинты и CSV-таблицы на диск, после чего память ОЗУ принудительно очищается через `del`. Система защищена от утечек памяти.
* **Цветовой датчик аномалий (GUI):** Встроенный 3D-интерфейс на Ursina с кнопками управления. Сферы динамически меняют цвет в зависимости от тропического натяжения $\lambda$ (Зеленый $\rightarrow$ Желтый предупреждающий $\rightarrow$ Красный эпицентр блокировки гена).
* **Трехуровневая научная верификация:** Автоматический расчет математической корреляции обратной матрицы [2.1], биологического RMSD-отклонения от снимков Крио-ЭМ [5.1] и углов физического излома полимера.
* **Авто-скриншоты аномалий:** Система самостоятельно делает снимки экрана в `.png` при фиксации критического перетяжения ДНК.

## 🛠️ Быстрый старт (Запуск в ОЗУ)
1. Установите зависимости:
   ```bash
   pip install cooler scikit-learn ursina pandas numpy
   ```
2. Запустите ядро симуляции:
   ```bash
   python src/real_dna_tropy.py
   ```
3. Для компиляции автономного EXE-приложения для биологов:
   ```bash
   pip install pyinstaller
   pyinstaller --onefile --windowed src/real_dna_tropy.py
   ```
# Release v1.0.0: Ultimate Synthesis of DNA Biophysics & Multi-Scale Genomics

This major update introduces a comprehensive thermodynamic, electrostatic, and multi-scale biochemical engine to the 3D Markov DNA processing pipeline. The system now seamlessly scales processing from core 3D spatial geometry down to 10,000 discrete nucleotide micro-zones, analyzing 10,000,000 real human genomic base pairs (Chr1: 50M-60M) in real-time.

## 🔬 New Biophysical & Environmental Features

### 1. Debye-Hückel Electrostatic Screening (Yukawa Potential)
* Replaced the static vacuum electrostatic model with a physically rigorous **Yukawa potential** simulation.
* Implemented a biological Debye length ($\lambda_D \approx 1.0$ nm) mimicking the realistic ionic concentration (NaCl/KCl salt solution) inside the host cell nucleus.
* Solutes and charges exponentially decay over spatial coordinates, neutralizing loop repulsions or tightly bound nodes in the 3D world.

### 2. Elastic Strain Energy (Bernoulli-Euler Beam Theory)
* Added real-time tracking of **Elastic Strain Energy** quantified in thermodynamic thermal units ($k_B T$).
* Calculated using the physical bending rigidity constant of double-stranded DNA ($B \approx 200 \text{ pN}\cdot\text{nm}^2$) relative to the Kirchhoff-Ilyukhin non-linear axial curvature.

### 3. Hydropathic Node Index (Hydrophobic Volumetric Pressure)
* Integrated a dynamic **Hydrophobic Matrix** analyzing water displacement during tight spatial packing.
* The pressure index updates based on local linear packing density correlated against the tropical distance matrix ($\lambda$).

---

## 🧬 Multi-Scale Genomic Processing (Rolling Windows)
The computational engine now scans nucleotide subsequences using continuous rolling windows across four distinct biological scales:
* **3-mer Scale (Point Rigidity):** Inherent local curvature utilizing the consensus Brukner tri-nucleotide bendability matrix.
* **5-mer Scale (Half-Turn Mechanics):** Scanning of 5-letter stacking energy windows to locate micro-hinges.
* **10-mer Scale (Macro-Periodicity):** Quantitative pitch correlation searching for 10-10.5 bp helical repeats, directly predicting nucleosome-binding affinity (histone core wrapping).
* **20-mer Scale (Regulatory CRISPR Profile):** Thermal and density validation of 20 bp target sequences for transcription factors and Cas9-guided systems.

---

## 🖥️ System Architecture & UI Optimization
* **Automated Self-Check Engine:** Embedded a 5-stage automated quality control block validation routine verifying data outputs against biological limits (Homo Sapiens mass distribution, GC skewed limits, mathematical boundary bounds).
* **Graphics Scale Enhancement:** decoupled internal physics arrays from presentation vectors by applying a $20.0\times$ 3D viewport spatial extension factor.
* **Aspect Optimization:** Reduced sphere node meshes and linear wire cylinder scales down to `0.015`, completely mitigating voxel clutter and rendering smooth macromolecular spatial chains with zero drop in FPS.

---

## 📊 Output Artifacts & Verification
* Comprehensive Macro Report (`dna_comprehensive_report_*.csv`) updated with multi-scale columns.
* Gigantic Detailed Micro Report (`dna_BIG_detailed_report_*.csv`) generating 10,000 rows of synchronized biochemical/biophysical parameters with Excel absolute formatting lock (`="val"` injection protection).
* Double-axis automated scientific deformation plots mapped to precise epoch timestamps.

## 📬 Автор
* **Александр Моисеенко** (`alekssan8183269-lang`) — Независимый исследователь / Independent Research.
