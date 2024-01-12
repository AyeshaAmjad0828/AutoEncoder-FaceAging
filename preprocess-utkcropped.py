import os
import shutil

def organize_images(input_folder, output_folder, young_min_age, young_max_age, old_min_age, old_max_age, young_target_count, old_target_count):
    # Create output folders if not exists
    young_folder = os.path.join(output_folder, "young")
    old_folder = os.path.join(output_folder, "old")

    os.makedirs(young_folder, exist_ok=True)
    os.makedirs(old_folder, exist_ok=True)

    # Iterate through images in the input folder
    image_files = sorted(os.listdir(input_folder))
    young_count = 0
    old_count = 0

    for image_file in image_files:
        # Extract age from the image file name
        age = int(image_file.split('_')[0])

        # Check if the age is within the specified range for the "young" folder
        if young_min_age <= age <= young_max_age and young_count < young_target_count:
            target_folder = young_folder
            young_count += 1
        # Check if the age is within the specified range for the "old" folder
        elif old_min_age <= age <= old_max_age and old_count < old_target_count:
            target_folder = old_folder
            old_count += 1
        else:
            continue

        # Copy the image to the target folder
        shutil.copy(os.path.join(input_folder, image_file), os.path.join(target_folder, image_file))

    print(f"Organized {young_count} images in 'young' folder and {old_count} images in 'old' folder.")

if __name__ == "__main__":
    input_folder = "C:\\Users\\ayesha.amjad\\Downloads\\archive\\utkcropped\\utkcropped"
    output_folder = "data/utk"

    young_min_age = 18
    young_max_age = 35
    young_target_count = 5

    old_min_age = 50
    old_max_age = 70
    old_target_count = 5

    organize_images(input_folder, output_folder, young_min_age, young_max_age, old_min_age, old_max_age, young_target_count, old_target_count)


