import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_personality_data(filepath: str, output_path: str = "personality_preprocessing_dataset.csv"):
    """
    Fungsi untuk melakukan preprocessing otomatis terhadap dataset kepribadian
    (Extrovert vs Introvert) dan menyimpan hasilnya ke file CSV.

    Params:
        filepath (str): Lokasi file CSV asli.
        output_path (str): Lokasi file hasil preprocessing (default: personality_preprocessing_dataset.csv).

    Returns:
        pd.DataFrame: Data hasil preprocessing, siap dilatih.
    """

    # Load data
    df = pd.read_csv(filepath)

    # 1. Hapus missing values
    df_clean = df.dropna()

    # 2. Hapus duplikat
    df_clean = df_clean.drop_duplicates()

    # 3. Normalisasi fitur numerik
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency']
    scaler = MinMaxScaler()
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    # 4. Label Encoding untuk fitur kategorikal
    cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le  # bisa digunakan untuk inverse transform jika perlu

    # 5. Simpan ke file CSV
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Data berhasil disimpan ke: {output_path}")

    return df_clean

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Extrovert vs Introvert Dataset")
    parser.add_argument("input_path", type=str, help="Path ke file dataset mentah (CSV)")
    parser.add_argument("--output", type=str, default="personality_preprocessing_dataset.csv", help="Path file output (opsional)")

    args = parser.parse_args()

    preprocess_personality_data(args.input_path, args.output)
