import pandas as pd
import glob
import os


def dataLoader(a, folder='.', suffix='.xlsx'):

    equipment_map = {
        'bot': 'Zprime_Socket_01',
        'pc': 'Zprime_Socket_02'
    }

    if a not in equipment_map:
        raise ValueError("Argument 'a' must be either 'bot' or 'pc'")
    
    file_list = glob.glob(os.path.join(folder, f'*{suffix}'))
    if not file_list:
        raise FileNotFoundError("No Excel files found in the specified folder.")

    dfs = [pd.read_excel(file) for file in file_list]
    df_concat = pd.concat(dfs, ignore_index=True)

    equipment_name = equipment_map[a]
    df_filtered = df_concat[df_concat['equipmentName'] == equipment_name].copy()
    
    return df_filtered

def save_path(filename, device='bot'):
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'figures', device)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)