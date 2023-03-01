import os
import html
import glob
import uuid
import hashlib
import requests
from tqdm import tqdm
from pdb import set_trace as st

eva3d_deepfashion_model = dict(file_url='https://drive.google.com/uc?id=1SYPjxnHz3XPRhTarx_Lw8SG_iz16QUMU',
                            alt_url='', file_size=160393221, file_md5='d0fae86edf76c52e94223bd3f39b2157',
                            file_path='checkpoint/512x256_deepfashion/volume_renderer/models_0420000.pt',)

eva3d_shhq_model = dict(file_url='https://drive.google.com/uc?id=1mSE9f9N7xTjSvsn9epTaDmFhNNIox2Hd',
                            alt_url='', file_size=160393221, file_md5='8b1a26134f3a832958addc71642b21ac',
                            file_path='checkpoint/512x256_shhq/volume_renderer/models_0740000.pt',)

eva3d_aist_model = dict(file_url='https://drive.google.com/uc?id=1jTzQRXVtlHXM1Zj9SBnGl2FZa_-aQFSK',
                            alt_url='', file_size=158403591, file_md5='2ccccb17b7571e6e96e0861e39f5d847',
                            file_path='checkpoint/256x256_aist/volume_renderer/models_0340000.pt',)

eva3d_ubcfashion_model = dict(file_url='https://drive.google.com/uc?id=1BCpc5tj8a1DDymYSYpTMo4YDYksUICE0',
                            alt_url='', file_size=160416931, file_md5='26bdd36d662c5bc28b6a597bceee4f03',
                            file_path='checkpoint/512x256_ubcfashion/volume_renderer/models_0620000.pt',)

def download_pretrained_models():
    print('Downloading EVA3D model pretrained on DeepFashion.')
    with requests.Session() as session:
        try:
            download_file(session, eva3d_deepfashion_model)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, eva3d_deepfashion_model, use_alt_url=True)
    print('Downloading EVA3D model pretrained on SHHQ.')
    with requests.Session() as session:
        try:
            download_file(session, eva3d_shhq_model)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, eva3d_shhq_model, use_alt_url=True)
    print('Downloading EVA3D model pretrained on UBCFashion.')
    with requests.Session() as session:
        try:
            download_file(session, eva3d_ubcfashion_model)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, eva3d_ubcfashion_model, use_alt_url=True)
    print('Downloading EVA3D model pretrained on AIST.')
    with requests.Session() as session:
        try:
            download_file(session, eva3d_aist_model)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, eva3d_aist_model, use_alt_url=True)
    

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
    download_pretrained_models()
