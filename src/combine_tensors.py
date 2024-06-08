print('Starting script')

import torch
import os
from multiprocessing import Pool

print('Starting script')
print('imported most')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print('Loaded libraries')

def load_tensor(file_path):
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def process_tensors(folder_path, batch_size=100000):
    files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
    file_paths = [os.path.join(folder_path, f) for f in files]
    total_files = len(file_paths)
    print(f"Total files to process: {total_files}")

    batches_processed = 0
    tensor_list = []

    # Use multiprocessing to load tensors in parallel
    with Pool() as pool:
        for i, tensor in enumerate(pool.imap(load_tensor, file_paths), 1):
            if tensor is not None:
                tensor_list.append(tensor)

            # Save batch when reaching batch size or at the end of the file list
            if len(tensor_list) >= batch_size or i == total_files:
                combined_tensor = torch.cat([t.unsqueeze(0) for t in tensor_list], 0)
                save_path = f"tensors/tensor_batch_{batches_processed}.pt"
                save_tensor_data(combined_tensor, files[(batches_processed * batch_size):i], save_path)
                batches_processed += 1

                # Clear the list and free up memory
                tensor_list = []
                torch.cuda.empty_cache()  # Call this if using CUDA

            if i % 10000 == 0:
                print(f"Processed {i} out of {total_files} files")

def save_tensor_data(tensor, names, save_path):
    data = {'combined_tensor': tensor, 'names': names}
    torch.save(data, save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    print('Loading tensors')
    folder_path = "ligands/"
    process_tensors(folder_path)
