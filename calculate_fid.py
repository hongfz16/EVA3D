from pytorch_fid import fid_score

def get_fid(our_path, target_path):
    return fid_score.calculate_fid_given_paths(
        [our_path, target_path],
        50, 'cuda', 2048, 8
    )
