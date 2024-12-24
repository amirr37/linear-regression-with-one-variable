import numpy as np
import matplotlib.pyplot as plt
import random
import time

random.seed(42)  # Ensure reproducibility
data = {
    "x": list(range(1, 101)),
    "y": [random.uniform(50, 150) for _ in range(100)]
}


def find_best_cost_function_params(x_series, y_series) -> tuple[float, float]:
    """

    :param x:  x dataset  - a list
    :param y:  y dataset - a list
    :return: w , b for f(x)=wx+b
    """
    m = len(x_series)
    tmp_w = 0
    tmp_b = 0
    change = 1
    learning_rate = 0.0001
    while change < 1000000:

        sum_w = 0
        sum_b = 0
        for x, y in zip(x_series, y_series):
            sum_w += ((tmp_w * x + tmp_b) - y) * x
            sum_b += ((tmp_w * x + tmp_b) - y)

        tmp_w = tmp_w - learning_rate * (1 / m) * sum_w
        tmp_b = tmp_b - learning_rate * (1 / m) * sum_b

        print("w = ", tmp_w)
        # print("b = ", tmp_b)
        change += 1

    return tmp_w, tmp_b


start = time.perf_counter()

best_w, best_b = find_best_cost_function_params(x_series=data['x'], y_series=data['y'])

# Create x values (e.g., from -10 to 10)
x = np.linspace(-5, 100, 100)  # 100 points between -10 and 10
y = best_w * x + best_b
end = time.perf_counter()

plt.plot(x, y, label="y = wx + b")

plt.scatter(x=data["x"], y=data["y"], s=50, c='red', label='actual Price')

print(f"Training time: {end - start:.4f} seconds")

plt.xlabel("x")
plt.ylabel("y")

plt.title("linear regression with one variable")
plt.legend()
plt.show()

