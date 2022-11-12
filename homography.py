import sys
import numpy as np


# class with all erquired methods
class Homography:
    # generate rows with one point
    def A_rows(self, X: list, X_prime: list) -> list:
        # x = X + [1]
        r1 = X + [0, 0, 0] + [-X[0] * X_prime[0], -X[1] * X_prime[0]]
        r2 = [0, 0, 0] + X + [-X[0] * X_prime[1], -X[1] * X_prime[1]]

        return [r1, r2]

    # generate matrix with n points
    def get_A(self, r2_points, projected_points):
        if len(r2_points) != len(projected_points):
            sys.exit("Provide same number of points in both spaces")

        A = list(
            map(lambda x: self.A_rows(x[0], x[1]), zip(r2_points, projected_points))
        )

        return np.array(A).reshape(-1, 8)

    # generate column with n points
    def get_B(self, X_prime):
        return np.array(X_prime).reshape(-1, 1)

    # gets H from given A and b using least squares
    def get_H(self, A, b):
        H = list(np.linalg.solve(A.T@A, A.T@b).reshape(-1))
        H.append(1)
        return np.array(H).reshape(3, 3)
