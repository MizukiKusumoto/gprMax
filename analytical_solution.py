import numpy as np
import math


# 出力先ディレクトリとファイル名
output_csv = "ana_filtered.csv"
output_csv_unfiltered = "ana.csv"
duration = 9.0e-9  # 時間ウィンドウ（秒）
domain_size = 3.0  # 計算領域のサイズ（メートル）
domain_num = 300  # グリッド数
rx = (0.2, 0.2, 0.2)  # 受信点の相対座標（x->y,y->z,z->x）

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

# フィルタ設定
OVERSAMPLE_FACTOR = 10  # オーバーサンプリング倍率
F_CUT = 2.0e9  # カットオフ周波数 [Hz]
FILTER_ORDER = 4  # バターワースフィルタの次数


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
        self.delay = -self.chi * 1.0
        self.dl = dl  # Length of the dipole
        self.f = self.gaussian
        self.fprime = self.gaussianprime
        self.fdoubleprime = self.gaussiandoubleprime

    def gaussian(self, time):
        """Calculates Gaussian waveform value at a specific time."""
        delay = time + self.delay
        return np.exp(-self.zeta * delay**2)

    def gaussianprime(self, time):
        """Calculates first derivative of Gaussian waveform at a specific time."""
        delay = time + self.delay
        return -2 * self.zeta * delay * np.exp(-self.zeta * delay**2)

    def gaussiandoubleprime(self, time):
        """Calculates second derivative of Gaussian waveform at a specific time."""
        delay = time + self.delay
        return (
            2
            * self.zeta
            * (2 * self.zeta * delay**2 - 1)
            * np.exp(-self.zeta * delay**2)
        )

    def gaussiantripleprime(self, time):
        """Calculates third derivative of Gaussian waveform at a specific time."""
        delay = time + self.delay
        return (
            -4
            * self.zeta**2
            * delay
            * (2 * self.zeta * delay**2 - 3)
            * np.exp(-self.zeta * delay**2)
        )

    def set_func(self):
        """Sets the function f(t) used in field calculations."""
        self.f = self.gaussianprime
        self.fprime = self.gaussiandoubleprime
        self.fdoubleprime = self.gaussiantripleprime
        # self.chi = 2**0.5 / self.freq
        # self.zeta = np.pi**2 * self.freq**2
        # self.delay = -self.chi * 1.0

    def Ex(self, time):
        """Calculates Ex component of the electric field at a specific time."""
        t = time - self.tau
        return (self.x * self.z / (4 * np.pi * self.epsilon * self.r**5)) * (
            3 * self.f(t)
            + 3 * self.tau * self.fprime(t)
            + self.tau**2 * self.fdoubleprime(t)
        )

    def Ey(self, time):
        """Calculates Ey component of the electric field at a specific time."""
        t = time - self.tau
        return (self.y * self.z / (4 * np.pi * self.epsilon * self.r**5)) * (
            3 * self.f(t)
            + 3 * self.tau * self.fprime(t)
            + self.tau**2 * self.fdoubleprime(t)
        )

    def Ez(self, time):
        """Calculates Ez component of the electric field at a specific time."""
        t = time - self.tau
        return (1 / (4 * np.pi * self.epsilon * self.r**5)) * (
            (2 * self.z**2 - (self.x**2 + self.y**2))
            * (self.f(t) + self.tau * self.fprime(t))
            - (self.x**2 + self.y**2) * self.tau**2 * self.fdoubleprime(t)
        )

    def Hx(self, time):
        """Calculates Hx component of the magnetic field at a specific time."""
        t = time - self.tau
        return -(self.y / (4 * np.pi * self.r**3)) * (
            self.fprime(t) + self.tau * self.fdoubleprime(t)
        )

    def Hy(self, time):
        """Calculates Hy component of the magnetic field at a specific time."""
        t = time - self.tau
        return (self.x / (4 * np.pi * self.r**3)) * (
            self.fprime(t) + self.tau * self.fdoubleprime(t)
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


def apply_butterworth_filter(data, dt, f_cut, order=4):
    """バターワースローパスフィルタを適用（周波数領域）"""
    N = len(data)
    freq = np.fft.fftfreq(N, dt)

    # バターワースの振幅応答
    H = 1.0 / np.sqrt(1.0 + (np.abs(freq) / f_cut) ** (2 * order))

    Y = np.fft.fft(data)
    Y_filtered = Y * H
    return np.real(np.fft.ifft(Y_filtered))


# 解析解を生成（オーバーサンプリング）
dt_fine = dt / OVERSAMPLE_FACTOR
iterations_fine = iterations * OVERSAMPLE_FACTOR

my_solution = AnalyticalSolution(rx[0], rx[1], rx[2], dl)
# my_solution.set_func()
fields_fine = my_solution.get_fields_time_series(iterations_fine, dt_fine)

# フィルタ適用
Ex_filtered = apply_butterworth_filter(fields_fine[:, 1], dt_fine, F_CUT, FILTER_ORDER)
Ey_filtered = apply_butterworth_filter(fields_fine[:, 2], dt_fine, F_CUT, FILTER_ORDER)
Ez_filtered = apply_butterworth_filter(fields_fine[:, 3], dt_fine, F_CUT, FILTER_ORDER)

# ダウンサンプリング
time = fields_fine[::OVERSAMPLE_FACTOR, 0][:iterations]
Ex = Ex_filtered[::OVERSAMPLE_FACTOR][:iterations]
Ey = Ey_filtered[::OVERSAMPLE_FACTOR][:iterations]
Ez = Ez_filtered[::OVERSAMPLE_FACTOR][:iterations]

# CSVとして保存
header = "time,Ex,Ey,Ez"
data_to_save = np.column_stack((time, Ex, Ey, Ez))
np.savetxt(output_csv, data_to_save, delimiter=",", header=header, comments="")

print(f"解析解を {output_csv} に保存しました。")

# フィルタなしバージョンも保存
fields_unfiltered = my_solution.get_fields_time_series(iterations, dt)
data_unfiltered = np.column_stack(
    (
        fields_unfiltered[:, 0],
        fields_unfiltered[:, 1],
        fields_unfiltered[:, 2],
        fields_unfiltered[:, 3],
    )
)
np.savetxt(
    output_csv_unfiltered, data_unfiltered, delimiter=",", header=header, comments=""
)

print(f"フィルタなし解析解を {output_csv_unfiltered} に保存しました。")
