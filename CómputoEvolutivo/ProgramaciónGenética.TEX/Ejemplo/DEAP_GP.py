import operator
import random
from deap import algorithms, base, creator, tools, gp
import numpy as np

# 1. Define the target function we want to approximate (e.g., x^2 + x)
def target_func(x):
    return x**2 + x

# 2. Create primitive set (building blocks)
pset = gp.PrimitiveSet("MAIN", arity=1)  # 1 input variable (x)
pset.addPrimitive(operator.add, 2, name="add")  # Addition
pset.addPrimitive(operator.mul, 2, name="mul")  # Multiplication
pset.addTerminal(1)                            # Constant 1
pset.renameArguments(ARG0="x")                 # Rename input to 'x'

# 3. Set up fitness and individual types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize error
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 4. Initialize toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 5. Define evaluation (mean squared error)
def evaluate(individual):
    func = gp.compile(individual, pset)
    points = [x/10.0 for x in range(-10, 11)]  # Test points from -1 to 1
    error = sum((func(x) - target_func(x))**2 for x in points)
    return error,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# 6. Run evolution
population = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

result, log = algorithms.eaSimple(
    population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, 
    stats=stats, halloffame=hof, verbose=True
)

# 7. Results
best_expr = str(hof[0])  # String representation of the best tree
print(f"\nBest expression: {best_expr}")

# Convert to readable math
readable_expr = best_expr.replace("add", "+").replace("mul", "*")
print(f"Human-readable: {readable_expr}")

# Plot results
import matplotlib.pyplot as plt
x = [i/10.0 for i in range(-10, 11)]
y_true = [target_func(xi) for xi in x]
y_pred = [gp.compile(hof[0], pset)(xi) for xi in x]

plt.plot(x, y_true, label="Target: $x^2 + x$")
plt.plot(x, y_pred, "--", label=f"Evolved: {readable_expr}")
plt.legend()
plt.show()
