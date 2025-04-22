from flask import Flask, render_template, request, session
from solver import solve_problem, generate_graph
from simplex import simplex_step_by_step
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_form', methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
        try:
            num_vars = int(request.form['num_vars'])
            num_constraints = int(request.form['num_constraints'])
            
            if not (2 <= num_vars <= 5) or not (2 <= num_constraints <= 5):
                raise ValueError("Número de variables y restricciones debe estar entre 2 y 5")
                
            return render_template('input_form.html', 
                                num_vars=num_vars, 
                                num_constraints=num_constraints)
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.form.to_dict()
        num_vars = int(data['num_vars'])
        
        objective = {
            'coefficients': [float(data[f'obj_var_{i+1}']) for i in range(num_vars)],
            'operation': data['operation']
        }
        
        constraints = []
        for i in range(int(data['num_constraints'])):
            constraints.append({
                'coefficients': [float(data[f'con_{i+1}_var_{j+1}']) for j in range(num_vars)],
                'operator': data[f'con_{i+1}_operator'],
                'rhs': float(data[f'con_{i+1}_rhs'])
            })
        
        # Guardar en sesión para los pasos simplex
        session['problem_data'] = {
            'objective': objective,
            'constraints': constraints,
            'num_vars': num_vars
        }
        
        solution = solve_problem(objective, constraints)
        graph = generate_graph(objective, constraints, solution) if num_vars == 2 else None
        
        return render_template('results.html',
                            solution=solution,
                            graph=graph,
                            num_vars=num_vars)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/simplex_steps')
def show_simplex_steps():
    try:
        problem_data = session.get('problem_data')
        if not problem_data:
            return render_template('error.html', error="Datos del problema no encontrados")
        
        steps = simplex_step_by_step(problem_data['objective'], 
                                   problem_data['constraints'])
        
        return render_template('simplex_steps.html', steps=steps)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)