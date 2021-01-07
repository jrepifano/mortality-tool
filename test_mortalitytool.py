import numpy as np
import mortalitytool


input_vec = np.array([2, 0, 2.43, 3.98, 2.28, 2.63, 100, 2, 39, 1.99, 16, 78])

d_test = mortalitytool.deterministic()
print(d_test.inference(input_vec))
print(d_test.get_explanation(input_vec))
s_test = mortalitytool.stochastic()
print(s_test.inference(input_vec))
print(s_test.get_explanation(input_vec))
