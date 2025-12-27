import os

COUNTER_FILE = "ref_counter.txt"
IMAGE_PATH = "images/fake/"
PREFIX = "fake_"

def read_counter():
    if not os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "w") as f:
            f.write("1")
        return 1

    with open(COUNTER_FILE, "r") as f:
        return int(f.read().strip())

def write_counter(value):
    with open(COUNTER_FILE, "w") as f:
        f.write(str(value))

def rename_images():
    counter = read_counter()

    for filename in sorted(os.listdir(IMAGE_PATH)):
        old_path = os.path.join(IMAGE_PATH, filename)

        # Skip non-files
        if not os.path.isfile(old_path):
            continue

        # Keep original extension
        ext = os.path.splitext(filename)[1]
        new_name = f"{PREFIX}{counter}{ext}"
        new_path = os.path.join(IMAGE_PATH, new_name)

        # Avoid overwriting files
        if os.path.exists(new_path):
            print(f"Skipping (exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

        counter += 1

    write_counter(counter)

if __name__ == "__main__":
    rename_images()
