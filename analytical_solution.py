import os
import numpy as np
from tests.analytical_solutions import hertzian_dipole_fs
import math


# 出力先ディレクトリとファイル名
output_csv = "ana.csv"
duration = 9.0e-9  # 時間ウィンドウ（秒）
domain_size = 3.0  # 計算領域のサイズ（メートル）
domain_num = 360  # グリッド数
rx = (0.8, 0.8, 0.8)  # 受信点の相対座標（x->y,y->z,z->x）

# 物理定数（C++と同じ値）
epsilon_0 = 8.8541878128e-12  # 真空の誘電率 [F/m]
mu_0 = 1.25663706212e-6  # 真空の透磁率 [H/m]
relative_permittivity = 1.0
relative_permeability = 1.0

# C++と同じ計算方法
permittivity = epsilon_0 * relative_permittivity
permeability = mu_0 * relative_permeability
c = 1.0 / np.sqrt(permittivity * permeability)

dt = domain_size / (domain_num * c * np.sqrt(3))  # 時間刻み（秒）
iterations = math.floor(duration / dt) + 1  # イテレーション数
dl = 0.001  # 導線長さ（メートル）


class AnalyticalSolution:
    """Class for calculating analytical solutions for specific scenarios."""

    def __init__(self, x, y, z, dl):
        self.x = x
        self.y = y
        self.z = z
        self.r = np.sqrt(x**2 + y**2 + z**2)
        self.epsilon = permittivity
        self.mu = permeability
        self.c = c  # Speed of light
        self.tau = self.r / self.c
        self.freq = 1e9  # Central frequency
        self.chi = 1 / self.freq
        self.zeta = 2 * np.pi**2 * self.freq**2
        self.delay = 0
        self.dl = dl  # Length of the dipole

    def gaussian(self, time):
        """Calculates Gaussian waveform value at a specific time."""
        delay = time - self.chi
        return np.exp(-self.zeta * delay**2)

    def gaussianprime(self, time):
        """Calculates first derivative of Gaussian waveform at a specific time."""
        delay = time - self.chi
        return -2 * self.zeta * delay * np.exp(-self.zeta * delay**2)

    def gaussiandoubleprime(self, time):
        """Calculates second derivative of Gaussian waveform at a specific time."""
        delay = time - self.chi
        return (
            2
            * self.zeta
            * (2 * self.zeta * delay**2 - 1)
            * np.exp(-self.zeta * delay**2)
        )

    def Ex(self, time):
        """Calculates Ex component of the electric field at a specific time."""
        t = time - self.tau
        return (self.x * self.z / (4 * np.pi * self.epsilon * self.r**5)) * (
            3 * self.gaussian(t)
            + 3 * self.tau * self.gaussianprime(t)
            + self.tau**2 * self.gaussiandoubleprime(t)
        )

    def Ey(self, time):
        """Calculates Ey component of the electric field at a specific time."""
        t = time - self.tau
        return (self.y * self.z / (4 * np.pi * self.epsilon * self.r**5)) * (
            3 * self.gaussian(t)
            + 3 * self.tau * self.gaussianprime(t)
            + self.tau**2 * self.gaussiandoubleprime(t)
        )

    def Ez(self, time):
        """Calculates Ez component of the electric field at a specific time."""
        t = time - self.tau
        return (1 / (4 * np.pi * self.epsilon * self.r**5)) * (
            (2 * self.z**2 - (self.x**2 + self.y**2))
            * (self.gaussian(t) + self.tau * self.gaussianprime(t))
            - (self.x**2 + self.y**2) * self.tau**2 * self.gaussiandoubleprime(t)
        )

    def Hx(self, time):
        """Calculates Hx component of the magnetic field at a specific time."""
        t = time - self.tau
        return -(self.y / (4 * np.pi * self.r**3)) * (
            self.gaussianprime(t) + self.tau * self.gaussiandoubleprime(t)
        )

    def Hy(self, time):
        """Calculates Hy component of the magnetic field at a specific time."""
        t = time - self.tau
        return (self.x / (4 * np.pi * self.r**3)) * (
            self.gaussianprime(t) + self.tau * self.gaussiandoubleprime(t)
        )

    def Hz(self, time):
        """Calculates Hz component of the magnetic field at a specific time."""
        return 0.0

    def get_fields_time_series(self, iterations, dt):
        """Generates time series of electric and magnetic fields at the observation point.

        Args:
            iterations (int): Number of time steps.
            dt (float): Time step size.
        """
        time_series = np.arange(0, iterations * dt, dt)
        fields = np.zeros((iterations, 7))  # 7 components: time, Ex, Ey, Ez, Hx, Hy, Hz

        for i, t in enumerate(time_series):
            fields[i, 0] = t
            fields[i, 1] = self.Ex(t) * self.dl
            fields[i, 2] = self.Ey(t) * self.dl
            fields[i, 3] = self.Ez(t) * self.dl
            fields[i, 4] = self.Hx(t) * self.dl
            fields[i, 5] = self.Hy(t) * self.dl
            fields[i, 6] = self.Hz(t) * self.dl

        return fields


my_solution = AnalyticalSolution(rx[0], rx[1], rx[2], dl)
fields = my_solution.get_fields_time_series(iterations, dt)

# time, Ex, Ey, Ez だけ抽出
time = fields[:, 0]
Ex = fields[:, 1]
Ey = fields[:, 2]
Ez = fields[:, 3]

# CSVとして保存
header = "time,Ex,Ey,Ez"
data_to_save = np.column_stack((time, Ex, Ey, Ez))
np.savetxt(output_csv, data_to_save, delimiter=",", header=header, comments="")

print(f"解析解を {output_csv} に保存しました。")
