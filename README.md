# Learning to Taste
Can machines learn how to taste? 🍷

This project is developed and tested on macOS and uses Python 3.11.

## Operating System
This software was developed and tested on macOS. While it is not specifically designed for other operating systems, it may work on Linux or Windows with minor modifications.

## Python Version
Python 3.11 is the programming language used for developing this software. This version of Python brings in several new features and optimizations that make the code efficient and easy to understand.

## Installation
Before running the software, you need to install the required Python libraries. You can install these dependencies using pip, Python's package installer. Here's how you can do it:

```bash
python3 -m pip install -r requirements.txt
```

This command will install all the necessary libraries listed in the requirements.txt file.

## Running the Program
Once you have installed the dependencies, you can run the program. The project consists of two main scripts: `run_experiments.py` and `run_taste_space_evaluation.py`.

To run the experiments, use the following command:

```bash
python3 -m run_experiments
```

This script will conduct a series of experiments based on predefined parameters and configurations. The results will be stored in a specified directory for further analysis.

To run the taste space evaluation, use the following command:

```bash
python3 -m run_taste_space_evaluation
```

This script evaluates the model's performance in predicting and understanding taste profiles. It generates a report that summarizes the model's accuracy and its understanding of the taste space.

## Output
The output of these scripts will be a series of files containing the results of the experiments and the taste space evaluation. You can analyze these files to understand how well the model performs, see patterns in the data, and gain insights into the world of tastes.

## Contribution
Contributions to this project are welcome. If you find a bug or want to propose a feature, feel free to open an issue or submit a pull request.
