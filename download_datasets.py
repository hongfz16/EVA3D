import os
import html
import glob
import uuid
import hashlib
import requests
from tqdm import tqdm
from pdb import set_trace as st

deepfashion_info = dict(file_url='https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBakxwRmctZjQ4bGpnYzFXU1dsMFhwWWZzWDBCTFE_ZT1vM3RWdmI/root/content',
                        alt_url='', file_size=4499345395, file_md5='cb3122bbf766ccc1cb68f6b61060c096',
                        file_path='datasets/DeepFashion.zip',)

def download_deepfashion():
    print('Downloading DeepFashion Datasets...')
    with requests.Session() as session:
        try:
            download_file(session, deepfashion_info)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, deepfashion_info, use_alt_url=True)
    os.system('unzip datasets/DeepFashion.zip -d datasets')
    os.system('rm datasets/DeepFashion.zip')


def download_file(session, file_spec, use_alt_url=False, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    if use_alt_url:
        file_url = file_spec['alt_url']
    else:
        file_url = file_spec['file_url']

    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    progress_bar = tqdm(total=file_spec['file_size'], unit='B', unit_scale=True)
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        progress_bar.reset()
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except Exception as e:
            # print(e)
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'confirm=t' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    progress_bar.close()

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

if __name__ == "__main__":
    download_deepfashion()
