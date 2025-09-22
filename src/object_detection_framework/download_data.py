import os
import zipfile



os.makedirs('./data/ai04-level1-project', exist_ok=True)
os.system('kaggle competitions download -c ai04-level1-project -p ./data')

with zipfile.ZipFile('data/ai04-level1-project.zip', 'r') as zip_ref:
    zip_ref.extractall('./data/ai04-level1-project')