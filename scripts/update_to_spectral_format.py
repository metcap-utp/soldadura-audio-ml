#!/usr/bin/env python3
"""
Script para actualizar los archivos de entrenamiento al formato JSON unificado de spectral-analysis.
"""

import re

def update_entrenar_py():
    with open('/home/luis/projects/vggish-backbone/entrenar.py', 'r') as f:
        content = f.read()
    
    # 1. Actualizar fold_results para incluir número de fold
    old_append = '''        metrics["time_seconds"] = round(fold_time, 2)
        
        fold_metrics.append(metrics)'''
    
    new_append = '''        metrics["time_seconds"] = round(fold_time, 2)
        metrics["fold"] = fold_idx
        
        fold_metrics.append(metrics)'''
    
    content = content.replace(old_append, new_append)
    
    # 2. Agregar blind evaluation al final (antes de guardar resultados)
    blind_evaluation_code = '''
    # ============= FASE 3: Evaluar en Blind Set =============
    print(f"\\n{'=' * 70}")
    print("FASE 3: EVALUACIÓN EN BLIND SET")
    print(f"{'=' * 70}")
    
    blind_csv = DURATION_DIR / f"blind_overlap_{OVERLAP_RATIO}.csv"
    if not blind_csv.exists():
        blind_csv = DURATION_DIR / "blind.csv"
    
    if blind_csv.exists():
        print(f"Cargando blind set desde: {blind_csv}")
        blind_df = pd.read_csv(blind_csv)
        
        # Extraer embeddings del blind set
        blind_embeddings = []
        for idx, row in blind_df.iterrows():
            if idx % 100 == 0:
                print(f"  Procesando blind {idx}/{len(blind_df)}...")
            emb = extract_vggish_embeddings_from_segment(
                row['Audio Path'], 
                int(row['Segment Index']), 
                SEGMENT_DURATION, 
                OVERLAP_SECONDS
            )
            blind_embeddings.append(emb)
        
        # Predecir con el ensemble
        blind_dataset = AudioDataset(
            blind_embeddings,
            [0] * len(blind_embeddings),
            [0] * len(blind_embeddings),
            [0] * len(blind_embeddings)
        )
        blind_loader = DataLoader(
            blind_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad
        )
        
        blind_preds = {"plate": [], "electrode": [], "current": []}
        with torch.no_grad():
            for embeddings, _, _, _ in blind_loader:
                pred_p, pred_e, pred_c = ensemble_predict(models, embeddings, device)
                blind_preds["plate"].extend(pred_p.cpu().numpy())
                blind_preds["electrode"].extend(pred_e.cpu().numpy())
                blind_preds["current"].extend(pred_c.cpu().numpy())
        
        # Decodificar predicciones
        y_true_plate = blind_df["Plate Thickness"].values
        y_true_electrode = blind_df["Electrode"].values
        y_true_current = blind_df["Type of Current"].values
        
        y_pred_plate = plate_encoder.inverse_transform(blind_preds["plate"])
        y_pred_electrode = electrode_encoder.inverse_transform(blind_preds["electrode"])
        y_pred_current = current_type_encoder.inverse_transform(blind_preds["current"])
        
        # Calcular métricas blind
        from sklearn.metrics import accuracy_score, f1_score
        
        blind_acc_plate = accuracy_score(y_true_plate, y_pred_plate)
        blind_acc_electrode = accuracy_score(y_true_electrode, y_pred_electrode)
        blind_acc_current = accuracy_score(y_true_current, y_pred_current)
        
        blind_f1_plate = f1_score(y_true_plate, y_pred_plate, average='weighted')
        blind_f1_electrode = f1_score(y_true_electrode, y_pred_electrode, average='weighted')
        blind_f1_current = f1_score(y_true_current, y_pred_current, average='weighted')
        
        # Exact match accuracy
        n_blind = len(y_true_plate)
        exact_matches = sum(
            1 for i in range(n_blind)
            if y_pred_plate[i] == y_true_plate[i]
            and y_pred_electrode[i] == y_true_electrode[i]
            and y_pred_current[i] == y_true_current[i]
        )
        exact_match_acc = exact_matches / n_blind
        hamming_acc = (blind_acc_plate + blind_acc_electrode + blind_acc_current) / 3
        
        blind_evaluation = {
            "plate": {
                "accuracy": round(blind_acc_plate, 4),
                "f1": round(blind_f1_plate, 4)
            },
            "electrode": {
                "accuracy": round(blind_acc_electrode, 4),
                "f1": round(blind_f1_electrode, 4)
            },
            "current": {
                "accuracy": round(blind_acc_current, 4),
                "f1": round(blind_f1_current, 4)
            },
            "global": {
                "exact_match": round(exact_match_acc, 4),
                "hamming_accuracy": round(hamming_acc, 4)
            }
        }
        
        print(f"\\nBlind Evaluation:")
        print(f"  Plate: Acc={blind_acc_plate:.4f}, F1={blind_f1_plate:.4f}")
        print(f"  Electrode: Acc={blind_acc_electrode:.4f}, F1={blind_f1_electrode:.4f}")
        print(f"  Current: Acc={blind_acc_current:.4f}, F1={blind_f1_current:.4f}")
        print(f"  Exact Match: {exact_match_acc:.4f}")
    else:
        print(f"No se encontró blind set en {blind_csv}")
        blind_evaluation = None
'''
    
    # Insertar antes de "Guardar resultados"
    insert_point = '    # Guardar resultados'
    content = content.replace(insert_point, blind_evaluation_code + '\n    # Guardar resultados')
    
    # 3. Actualizar new_entry para incluir blind_evaluation
    old_entry_end = '''        "training_history": all_fold_histories,
    }'''
    
    new_entry_end = '''        "training_history": all_fold_histories,
        "blind_evaluation": blind_evaluation,
    }'''
    
    content = content.replace(old_entry_end, new_entry_end)
    
    # 4. Actualizar promedios individuales
    old_avg = '''    avg_acc_p = np.mean([m["acc_plate"] for m in fold_metrics])
    avg_acc_e = np.mean([m["acc_electrode"] for m in fold_metrics])
    avg_acc_c = np.mean([m["acc_current"] for m in fold_metrics])'''
    
    new_avg = '''    avg_acc_p = np.mean([m["accuracy_plate"] for m in fold_metrics])
    avg_acc_e = np.mean([m["accuracy_electrode"] for m in fold_metrics])
    avg_acc_c = np.mean([m["accuracy_current"] for m in fold_metrics])'''
    
    content = content.replace(old_avg, new_avg)
    
    with open('/home/luis/projects/vggish-backbone/entrenar.py', 'w') as f:
        f.write(content)
    
    print("✓ entrenar.py actualizado")


if __name__ == "__main__":
    update_entrenar_py()
