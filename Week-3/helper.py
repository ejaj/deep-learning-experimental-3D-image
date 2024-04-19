from pathlib import Path


def get_patient_data(dataset_path: Path, first_id: int, last_id: int):
    dataset = []
    for i in range(first_id, last_id + 1):
        patient_id = f'patient{str(i).zfill(3)}'
        tmp_path = dataset_path / patient_id
        for path_frame in tmp_path.glob("*frame[0-9][0-9].nii.gz"):
            dataset.append({'image': str(path_frame),
                            'label': str(path_frame).replace(".nii.gz", "_gt.nii.gz"),
                            'id': path_frame.name.split(".")[0]})
    return dataset


DATA_PATH = "data/ACDC17"
train_files = get_patient_data(dataset_path=Path(DATA_PATH), first_id=1, last_id=14)
val_files = get_patient_data(dataset_path=Path(DATA_PATH), first_id=15, last_id=16)
test_files = get_patient_data(dataset_path=Path(DATA_PATH), first_id=17, last_id=20)

print(train_files)