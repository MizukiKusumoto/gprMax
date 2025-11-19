import h5py
import numpy as np
import pandas as pd
import sys

def gprmax_to_csv(output_file, csv_file, n, rx_number=1):
    """
    gprMaxのHDF5出力ファイルをCSVに変換
    
    Parameters:
    -----------
    output_file : str
        gprMaxの出力ファイル名（.out）
    csv_file : str
        出力CSVファイル名
    rx_number : int
        受信点番号（デフォルト: 1）
    """
    
    # HDF5ファイルを開く
    with h5py.File(output_file, 'r') as f:
        # 時間ステップを取得
        dt = f.attrs['dt']
        
        # 受信点のパスを構築
        rx_path = f'/rxs/rx{rx_number}'
        
        # 電場成分を取得
        Ex = f[f'{rx_path}/Ex'][:]
        Ey = f[f'{rx_path}/Ey'][:]
        Ez = f[f'{rx_path}/Ez'][:]
        
        # 時間配列を作成
        n_steps = len(Ex)
        time = np.arange(n_steps) * dt
    
    # DataFrameを作成
    df = pd.DataFrame({
        'time': time,
        'Ex': Ex,
        'Ey': Ey,
        'Ez': Ez
    })
    
    # CSVに保存
    df = df[0::n]
    df.to_csv(csv_file+f'_0_{n}.csv', index=False)
    print(f"データを {csv_file}_0_{n}.csv に保存しました")
    print(f"時間ステップ: {dt:.3e} s")
    print(f"データ点数: {n_steps}")

if __name__ == "__main__":
    n = int(sys.argv[1])
    gprmax_to_csv('test.out', 'gprMax', n)