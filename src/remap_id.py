import os 

def remap_id(input_dir, output_dir, id):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), "r") as f_in, open(os.path.join(output_dir, file), "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                parts[0] = str(id)
                f_out.write(" ".join(parts) + "\n")

# remap_id('src/train/labels_1', 'src/train/labels', 80)
# remap_id('src/valid/labels_1', 'src/valid/labels', 80)

