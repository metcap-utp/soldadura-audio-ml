# Instrucciones resumidas para diagramas (PlotNeuralNet)

## Objetivo
- Diagrama limpio y publicable, sin texto superpuesto.
- Reflejar exactamente la arquitectura del modelo (modelo.py es la fuente de verdad).

## Reglas de rotulado
- Evitar nodos sueltos. Todo el texto va dentro de cada capa como `caption`.
- Usar `\parbox{<ancho>}{\centering ...}` para captions multilinea (evita errores de pgfkeys).
- Formato de dimensiones: `a×b×c` (no usar flechas). Ejemplos:
  - Conv1D: `out×in×k` (p. ej. `256×128×5`).
  - FC: `in×out` (p. ej. `1024×256`).
  - Heads: `emb×clases` (p. ej. `256×3`).
  - VGGish: `1×1×128`.
- Incluir activaciones solo si aporta: `BN + ReLU`.
- Evitar comas en captions dentro de `\pic` (si se necesitan, usar `{,}`).

## Estilo recomendado
- `\small\centering` para el titulo de capa y `\footnotesize` para dimensiones.
- Mantener el mismo ancho de `\parbox` para capas similares.
- Nombres claros y consistentes con el modelo (p. ej. `Espesor`, `Electrodo`, `Corriente`).

## Flujo de trabajo
1. Editar `arquitectura/smaw_xvector.py` (o script equivalente).
2. Generar `.tex` con el script.
3. Compilar:
   - `pdflatex -interaction=nonstopmode <archivo>.tex`
4. Convertir a PNG:
   - `pdftoppm <archivo>.pdf <archivo> -png -r 200`

## Errores comunes
- PDF en blanco: revisar que no haya `\n` dentro de argumentos TikZ.
- Error de pgfkeys: falta de `\parbox` o comas sin escapar.
- Texto desordenado: no usar `\node` externos.
