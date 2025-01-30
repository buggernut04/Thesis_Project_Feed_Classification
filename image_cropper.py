import cv2
import os

def cut_image(input_path, output_folder, division_size):
    # Load the image
    image = cv2.imread(input_path)

    if image is None:
        print(f"Error: Unable to load the image from {input_path}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Divide the image into parts
    for y in range(0, image.shape[0], division_size):
        for x in range(0, image.shape[1], division_size):
            divided_part = image[y:y+division_size, x:x+division_size]
            part_filename = f'part_{x}_{y}.jpg'
            part_path = os.path.join(output_folder, part_filename)
            cv2.imwrite(part_path, divided_part)

def delete_images_with_wrong_size(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            # Load the image
            image = cv2.imread(filepath)
            if image is not None:
                # Check if the image size is 300x300
                if image.shape[:2] != (300, 300):
                    # Delete the image
                    os.remove(filepath)
            else:
                print(f"Error: Unable to load {filename}")

if __name__ == "__main__":
    for i in range(1, 6):
        input_image_path = rf'C:\Users\hp\Documents\VSU Files\Fourth Year\Thesis\Program\Dataset Images\Rice Bran\Raw\Pure\P_Raw_RB ({i}).jpg'
        output_folder_path = rf'C:\Users\hp\Documents\VSU Files\Fourth Year\Thesis\Program\Dataset Images\Rice Bran\Cropped\Pure\cropped_P_ricebran_{i}'

        os.makedirs(output_folder_path, exist_ok=True)
        division_size = 300

        cut_image(input_image_path, output_folder_path, division_size)
        delete_images_with_wrong_size(output_folder_path)

    # input_image_path = rf'C:\Users\hp\Documents\VSU Files\Fourth Year\Thesis\Program\Dataset Images\Rice Bran\Raw\Adulterated\A_Raw_RB (4).jpg'
    # output_folder_path = rf'C:\Users\hp\Documents\VSU Files\Fourth Year\Thesis\Program\Dataset Images\Rice Bran\Cropped\Adulterated\cropped_A_ricebran_4'

    # os.makedirs(output_folder_path, exist_ok=True)
    # division_size = 300

    # cut_image(input_image_path, output_folder_path, division_size)
    # delete_images_with_wrong_size(output_folder_path)
    