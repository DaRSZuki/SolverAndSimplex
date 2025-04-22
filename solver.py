import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def solve_problem(objective, constraints):
    """Resuelve el problema de programación lineal"""
    try:
        # Preparar los coeficientes para scipy
        c = np.array(objective['coefficients'])
        if objective['operation'] == 'maximize':
            c = -c  # linprog solo minimiza
        
        # Preparar las restricciones
        A = []
        b = []
        bounds = [(0, None) for _ in range(len(c))]  # Variables no negativas
        
        for constraint in constraints:
            A.append(constraint['coefficients'])
            b.append(constraint['rhs'])
        
        A = np.array(A)
        b = np.array(b)
        
        # Manejar diferentes tipos de restricciones
        eq_constraints = [i for i, con in enumerate(constraints) if con['operator'] == '==']
        ineq_constraints = [i for i, con in enumerate(constraints) if con['operator'] != '==']
        
        A_ub = A[ineq_constraints] if ineq_constraints else None
        b_ub = b[ineq_constraints] if ineq_constraints else None
        A_eq = A[eq_constraints] if eq_constraints else None
        b_eq = b[eq_constraints] if eq_constraints else None
        
        # Resolver el problema
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            return {
                'optimal_value': -res.fun if objective['operation'] == 'maximize' else res.fun,
                'variables': res.x.tolist(),
                'status': 'Óptimo encontrado',
                'message': res.message
            }
        else:
            return {
                'status': 'No se pudo encontrar solución óptima',
                'message': res.message
            }
    
    except Exception as e:
        return {
            'status': 'Error en la solución',
            'message': str(e)
        }

def generate_graph(objective, constraints, solution):
    """Genera un gráfico para problemas con 2 variables"""
    try:
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calcular rangos automáticamente
        x_max, y_max = 0, 0
        for con in constraints:
            a, b = con['coefficients']
            rhs = con['rhs']
            
            if a != 0:
                x_max = max(x_max, rhs/a * 1.2)
            if b != 0:
                y_max = max(y_max, rhs/b * 1.2)
        
        x_max = max(x_max, 10)
        y_max = max(y_max, 10)
        
        x = np.linspace(0, x_max, 400)
        
        # Graficar restricciones
        for i, con in enumerate(constraints):
            a, b = con['coefficients']
            rhs = con['rhs']
            
            if b != 0:
                y = (rhs - a * x) / b
                mask = (y >= 0) & (y <= y_max)
                label = f'{a}x1 + {b}x2 {con["operator"]} {rhs}'
                
                if con['operator'] == '<=':
                    ax.plot(x[mask], y[mask], color=f'C{i}', label=label)
                    ax.fill_between(x[mask], 0, y[mask], color=f'C{i}', alpha=0.1)
                elif con['operator'] == '>=':
                    ax.plot(x[mask], y[mask], color=f'C{i}', label=label)
                    ax.fill_between(x[mask], y[mask], y_max, color=f'C{i}', alpha=0.1)
                else:
                    ax.plot(x[mask], y[mask], color=f'C{i}', linestyle='--', label=label)
            else:
                x_val = rhs / a
                ax.axvline(x=x_val, color=f'C{i}', label=f'{a}x1 {con["operator"]} {rhs}')
                if con['operator'] == '<=':
                    ax.fill_betweenx([0, y_max], 0, x_val, color=f'C{i}', alpha=0.1)
                elif con['operator'] == '>=':
                    ax.fill_betweenx([0, y_max], x_val, x_max, color=f'C{i}', alpha=0.1)
        
        # Graficar solución óptima
        if solution and 'variables' in solution and len(solution['variables']) == 2:
            x_opt, y_opt = solution['variables']
            ax.scatter(x_opt, y_opt, color='red', s=100, marker='*', 
                      label=f'Solución óptima ({x_opt:.1f}, {y_opt:.1f})')
        
        # Configuración del gráfico
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title('Método Gráfico', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Convertir a imagen
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    
    except Exception as e:
        print(f"Error al generar gráfico: {str(e)}")
        return None