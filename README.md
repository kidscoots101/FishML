## FishML

### Enhancing Fish Disease Diagnosis through Machine Learning

Through this solution, we intend to benefit 2 main entities: fish owners and lastly the fishes themselves.

We want to be able to speed up the processes of identifying diseases in fish such that it is able to seek treatment much faster and relieve the stress for fish owners to constantly take care after their fishes. 

## Python Packages Used
- `os` (standard library)
- `numpy`
- `streamlit`
- `tensorflow`
- `PIL` (Python Imaging Library, installed via `Pillow`)

## Installation
`pip install numpy streamlit tensorflow Pillow` -> This installs ALL the packages required for our program.

### NOTE
In our program there's a specific segment which reuqires customistation on your end. As the path to our datasets will be different from yours, it will require some changing.
On line 15 it says:
`base_dir = '/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/Dataset.csv'`
You should replace it with:
`base_dir = '/Users/{yourusername}/path/to/fish-disease-diagnosis-py/Dataset.csv'`
^ Change it to the path in which our program is found in your computer.

