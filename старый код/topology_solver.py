from pyknotid.spacecurves import Knot

def analyze_dna_knot(coordinates):
    """
    Принимает массив 3D-точек реальной ДНК и определяет тип узла.
    """
    if len(coordinates) < 4:
        return "Нить слишком короткая для узла"
        
    # Инициализируем объект узла в pyknotid
    dna_loop = Knot(coordinates)
    
    # Считаем инварианты (математический хэш узла)
    # Например, детерминант узла (для трилистника он равен 3)
    det = dna_loop.determinant() 
    
    # Пытаемся автоматически определить тип узла по базе данных
    knot_type = dna_loop.identify()
    
    return {
        "determinant": det,
        "predicted_type": knot_type,
        "crossings": dna_loop.num_crossings() # Число пересечений в проекции
    }
