# Define the substrings to look for
substrs = [
    "Using model: ",
    "Human kernel: ",
    "Combined: ",
    "Machine kernel: ",
    "TAR: "
]

with open("output2.txt", "r") as f:
    for line in f:
        # If the line starts with any of the substrings, print it
        if any(line.startswith(substr) for substr in substrs):
            print(line.strip())  # strip() removes trailing newline