import os
import shutil

def rename_files_in_folder(source_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the folder name (for naming files)
    folder_name = os.path.basename(source_folder)

    # Get all the files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Loop through the files and rename them
    for idx, file in enumerate(files, start=1):
        # Construct the new file name
        new_name = f"{folder_name}_{idx}{os.path.splitext(file)[-1]}"

        # Source and destination paths
        src_file_path = os.path.join(source_folder, file)
        dest_file_path = os.path.join(output_folder, new_name)

        # Copy the file to the new location with the new name
        shutil.copy(src_file_path, dest_file_path)
        print(f"Renamed: {file} -> {new_name}")

# Example usage
source_folder = "D:/计算机视觉学习/deep-learning-for-image-processing-master/data_set/cuttings/val/Dark-gray-mudstone"  # Replace with your folder path
#output_folder = "D:/计算机视觉学习/deep-learning-for-image-processing-master/data_set/cuttings/train/renamed"  # Replace with your desired output folder path

rename_files_in_folder(source_folder, source_folder)
