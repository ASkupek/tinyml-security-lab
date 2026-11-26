from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(waveforms,labels, seed=42):
    X = np.asarray(waveforms)
    y = np.asarray(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )

    for name, y_part in [
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:
        counts = np.bincount(y_part, minlength=2)
        print(f"{name}: {len(y_part)} Together, {counts[0]} normal, {counts[1]} anomalies")
    return X_train, X_val, X_test, y_train, y_val, y_test
