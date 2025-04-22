import numpy as np
from fractions import Fraction

def simplex_step_by_step(objective, constraints):
    """Implementación completa del Método Simplex paso a paso"""
    steps = []
    steps.append("=== INICIO DEL MÉTODO SIMPLEX ===")

    # Configuración inicial
    c = np.array(objective['coefficients'], dtype=float)
    is_maximization = (objective['operation'] == 'maximize')
    num_vars = len(c)
    num_constraints = len(constraints)
    
    # Matriz de restricciones
    A = np.array([con['coefficients'] for con in constraints], dtype=float)
    b = np.array([con['rhs'] for con in constraints], dtype=float)
    operators = [con['operator'] for con in constraints]
    
    # Variables adicionales
    slack_vars = 0
    artificial_vars = 0
    var_names = [f'x{i+1}' for i in range(num_vars)]
    
    # Extender la matriz A y el vector c
    A_ext = A.copy()
    c_ext = c.copy()
    
    for i in range(num_constraints):
        if operators[i] == '<=':
            # Agregar variable de holgura
            new_col = np.zeros((num_constraints, 1))
            new_col[i] = 1
            A_ext = np.hstack([A_ext, new_col])
            c_ext = np.append(c_ext, 0)
            var_names.append(f's{slack_vars+1}')
            slack_vars += 1
        
        elif operators[i] == '>=':
            # Restar variable de exceso y agregar artificial
            new_col = np.zeros((num_constraints, 1))
            new_col[i] = -1
            A_ext = np.hstack([A_ext, new_col])
            c_ext = np.append(c_ext, 0)
            var_names.append(f'e{slack_vars+1}')
            slack_vars += 1
            
            new_col = np.zeros((num_constraints, 1))
            new_col[i] = 1
            A_ext = np.hstack([A_ext, new_col])
            c_ext = np.append(c_ext, 1e6)  # Método de la M grande
            var_names.append(f'a{artificial_vars+1}')
            artificial_vars += 1
        
        else:  # '=='
            # Agregar variable artificial
            new_col = np.zeros((num_constraints, 1))
            new_col[i] = 1
            A_ext = np.hstack([A_ext, new_col])
            c_ext = np.append(c_ext, 1e6)
            var_names.append(f'a{artificial_vars+1}')
            artificial_vars += 1

    # Línea corregida (paréntesis cerrado)
    steps.append("Sistema en forma estándar:")
    steps.append(f"• Función objetivo: {' + '.join([f'{c_ext[i]}{var_names[i]}' for i in range(len(c_ext))])}")
    steps.append(f"• Variables: {', '.join(var_names)}")
    steps.append(f"• Variables artificiales añadidas: {artificial_vars}")

    # Identificar variables básicas iniciales
    basis = []
    basis_indices = []
    for j in range(num_vars, len(var_names)):
        col = A_ext[:, j]
        if sum(col == 1) == 1 and sum(col != 0) == 1:
            row = np.where(col == 1)[0][0]
            if row not in [idx for idx, _ in basis_indices]:
                basis.append(var_names[j])
                basis_indices.append((row, j))

    # Ordenar por fila
    basis_indices.sort()
    basis = [var_names[j] for (_, j) in basis_indices]
    basis_indices = [j for (_, j) in basis_indices]

    # Crear tabla simplex
    tableau = np.hstack([A_ext, b.reshape(-1, 1)])
    z_row = np.hstack([-c_ext if is_maximization else c_ext, 0])
    tableau = np.vstack([tableau, z_row])

    steps.append("\nTabla inicial:")
    steps.append(format_table(tableau, var_names + ['RHS'], basis))

    # Iteraciones del simplex
    iteration = 1
    optimal = False
    unbounded = False

    while not optimal and iteration <= 100:
        steps.append(f"\nIteración {iteration}:")
        
        # Condición de optimalidad
        z_row = tableau[-1, :-1]
        if is_maximization:
            optimal = np.all(z_row >= -1e-6)
        else:
            optimal = np.all(z_row <= 1e-6)

        if optimal:
            steps.append("¡Solución óptima encontrada!")
            break

        # Selección de variable entrante
        if is_maximization:
            entering_col = np.argmin(z_row)
        else:
            entering_col = np.argmax(z_row)
        steps.append(f"Variable entrante: {var_names[entering_col]}")

        # Selección de variable saliente
        ratios = []
        for i in range(num_constraints):
            if tableau[i, entering_col] > 1e-6:
                ratios.append(tableau[i, -1] / tableau[i, entering_col])
            else:
                ratios.append(np.inf)

        if all(np.isinf(ratios)):
            unbounded = True
            steps.append("¡Problema no acotado!")
            break

        leaving_row = np.argmin(ratios)
        steps.append(f"Variable saliente: {basis[leaving_row]}")
        steps.append(f"Elemento pivote: {tableau[leaving_row, entering_col]:.4f}")

        # Actualizar base
        basis[leaving_row] = var_names[entering_col]
        basis_indices[leaving_row] = entering_col

        # Operaciones de fila
        pivot_val = tableau[leaving_row, entering_col]
        tableau[leaving_row, :] /= pivot_val

        for i in range(num_constraints + 1):
            if i != leaving_row:
                multiplier = tableau[i, entering_col]
                tableau[i, :] -= multiplier * tableau[leaving_row, :]

        steps.append("Tabla actualizada:")
        steps.append(format_table(tableau, var_names + ['RHS'], basis))
        iteration += 1

    # Solución final
    steps.append("\n=== SOLUCIÓN FINAL ===")
    
    if unbounded:
        steps.append("El problema es no acotado")
        return steps

    solution = np.zeros(num_vars)
    for i, var_idx in enumerate(basis_indices):
        if var_idx < num_vars:
            solution[var_idx] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    if is_maximization:
        optimal_value *= -1

    steps.append("Valores de las variables:")
    for i in range(num_vars):
        steps.append(f"• {var_names[i]} = {solution[i]:.4f}")
    
    steps.append(f"\nValor óptimo: {optimal_value:.4f}")
    return steps

def format_table(tableau, headers, basis):
    """Formatea la tabla simplex para visualización"""
    def format_value(val):
        if abs(val) < 1e-6:
            return "    0    "
        try:
            f = Fraction(val).limit_denominator(100)
            return f"{str(f):>8}" if f.denominator != 1 else f"{f.numerator:>8}"
        except:
            return f"{val:>8.2f}"

    lines = []
    num_rows, num_cols = tableau.shape
    
    # Encabezados
    header_line = "Básica | " + " | ".join(f"{h:>8}" for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Filas
    for i in range(num_rows - 1):
        row = [format_value(val) for val in tableau[i, :]]
        lines.append(f"{basis[i]:>6} | " + " | ".join(row))
    
    # Fila Z
    lines.append("-" * len(header_line))
    z_row = [format_value(val) for val in tableau[-1, :]]
    lines.append("   Z   | " + " | ".join(z_row))
    
    return "\n".join(lines)