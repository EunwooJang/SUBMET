from main import *

run_list = [0, 1, 2, 11, 20, 30, 40, 50, 56]
board = 10

for run in run_list:
    plot_extra_header(run, board)
    print(run)

# Configuration
file_path = "files/images/extra header/"
pos_count = 32
images_per_row = len(run_list)
images_per_column = 4
grid_size = images_per_row * images_per_column  # Total images per large image


# Function to get image file names
def get_image_file(poss, runs):
    return f"Extra header Pos.{poss} Run.{runs} Board.{board}.jpg"


# Create the large images
for i in range(pos_count // images_per_column):
    large_image = Image.new('RGB', (0, 0))  # Placeholder to calculate final size later
    pos_start = i * images_per_column
    pos_end = min((i + 1) * images_per_column, pos_count)

    row_images = []  # Holds rows of images

    for pos in range(pos_start, pos_end):
        row = []
        for run in run_list:
            img_file = get_image_file(pos, run)
            img_path = os.path.join(file_path, img_file)

            try:
                img = Image.open(img_path)
                row.append(img)
            except FileNotFoundError:
                print(f"Image not found: {img_path}. Skipping.")

        if row:
            widths, heights = zip(*(img.size for img in row))
            total_width = sum(widths)
            max_height = max(heights)
            combined_row = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in row:
                combined_row.paste(img, (x_offset, 0))
                x_offset += img.width

            row_images.append(combined_row)

    if row_images:
        max_width = max(row.width for row in row_images)
        total_height = sum(row.height for row in row_images)
        large_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for row in row_images:
            large_image.paste(row, (0, y_offset))
            y_offset += row.height

        output_path = os.path.join(file_path, f"combined_{i + 1}.jpg")
        large_image.save(output_path)
        print(f"Saved: {output_path}")
