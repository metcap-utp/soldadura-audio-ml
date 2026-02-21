#!/usr/bin/env python3
"""
Actualizar entrenar_ecapa.py y entrenar_feedforward.py al formato spectral-analysis.
"""

def update_ecapa():
    with open('/home/luis/projects/vggish-backbone/entrenar_ecapa.py', 'r') as f:
        content = f.read()
    
    # 1. Cambiar nombre de métricas
    content = content.replace('"acc_plate":', '"accuracy_plate":')
    content = content.replace('"acc_electrode":', '"accuracy_electrode":')
    content = content.replace('"acc_current":', '"accuracy_current":')
    
    # 2. Agregar campo fold
    old_line = 'metrics["time_seconds"] = round(fold_time, 2)'
    new_line = 'metrics["time_seconds"] = round(fold_time, 2)\n        metrics["fold"] = fold_idx'
    content = content.replace(old_line, new_line)
    
    # 3. Actualizar referencias en promedios
    content = content.replace('[m["acc_plate"]', '[m["accuracy_plate"]')
    content = content.replace('[m["acc_electrode"]', '[m["accuracy_electrode"]')
    content = content.replace('[m["acc_current"]', '[m["accuracy_current"]')
    
    with open('/home/luis/projects/vggish-backbone/entrenar_ecapa.py', 'w') as f:
        f.write(content)
    print("✓ entrenar_ecapa.py actualizado")


def update_feedforward():
    with open('/home/luis/projects/vggish-backbone/entrenar_feedforward.py', 'r') as f:
        content = f.read()
    
    # 1. Cambiar nombre de métricas
    content = content.replace('"acc_plate":', '"accuracy_plate":')
    content = content.replace('"acc_electrode":', '"accuracy_electrode":')
    content = content.replace('"acc_current":', '"accuracy_current":')
    
    # 2. Agregar campo fold
    old_line = 'metrics["time_seconds"] = round(fold_time, 2)'
    new_line = 'metrics["time_seconds"] = round(fold_time, 2)\n        metrics["fold"] = fold_idx'
    content = content.replace(old_line, new_line)
    
    # 3. Actualizar referencias en promedios
    content = content.replace('[m["acc_plate"]', '[m["accuracy_plate"]')
    content = content.replace('[m["acc_electrode"]', '[m["accuracy_electrode"]')
    content = content.replace('[m["acc_current"]', '[m["accuracy_current"]')
    
    with open('/home/luis/projects/vggish-backbone/entrenar_feedforward.py', 'w') as f:
        f.write(content)
    print("✓ entrenar_feedforward.py actualizado")


if __name__ == "__main__":
    update_ecapa()
    update_feedforward()
