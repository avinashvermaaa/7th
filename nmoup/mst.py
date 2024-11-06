def bisection_method(func, a, b, tolerance=1e-6, max_iterations=100):
  if func(a) * func(b) >= 0:
    print("Function values at the interval bounds must have opposite signs.")
    return None

  for _ in range(max_iterations):
    c = (a + b) / 2
    if func(c) == 0 or (b - a) / 2 < tolerance:
      return c
    elif func(c) * func(a) < 0:
      b = c
    else:
      a = c

  print("Maximum iterations reached without finding a root within the tolerance.")
  return None

def f(x):
  return x**2 - 4*x + 3

root = bisection_method(f, 2, 1e16)

if root is not None:
  print("Root:", root)
