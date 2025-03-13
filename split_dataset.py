import json
import random

# Load the dataset from the JSON file
with open("/work/pi_wenlongzhao_umass_edu/27/atifabedeen/CodeXGLUE/Text-Code/NL-code-search-WebQuery/data/cosqa_train.json", "r") as infile:
    data = json.load(infile)

# Print the total number of entries
print("Total entries:", len(data))

# Randomly shuffle the data
random.shuffle(data)

# Split the data into half
half_length = len(data) // 2
half_data = data[:half_length]

# Save the halved dataset to a new JSON file
with open("cosqa_train_half.json", "w") as outfile:
    json.dump(half_data, outfile, indent=2)

print(f"Saved halved dataset with {len(half_data)} entries to 'cosqa_half.json'")
