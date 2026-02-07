import os, shutil
from PIL import Image
from sklearn.model_selection import train_test_split

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def main():
    data_dir = '/kaggle/input/microsoft-catsvsdogs-dataset'
    raw_dir = os.path.join(data_dir, 'PetImages')
    split_dir = '/kaggle/working/splits'
    classes = ['Cat', 'Dog']
    images, labels = [], []
    
    
    for i, cls in enumerate(classes):
        for f in os.listdir(os.path.join(raw_dir, cls)):
            p = os.path.join(raw_dir, cls, f)
            if is_valid_image(p):
                images.append(p)
                labels.append(i)
    
    
    tr_x, tmp_x, tr_y, tmp_y = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    val_x, te_x, val_y, te_y = train_test_split(tmp_x, tmp_y, test_size=0.5, stratify=tmp_y, random_state=42)
    
    
    splits = {'train': (tr_x, tr_y), 'val': (val_x, val_y), 'test': (te_x, te_y)}
    for split, (xs, ys) in splits.items():
        for p, y in zip(xs, ys):
            out = os.path.join(split_dir, split, classes[y])
            os.makedirs(out, exist_ok=True)
            shutil.copy(p, out)
    print('Dataset split done.')




if __name__ == '__main__':
    main()