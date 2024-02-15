## FishML

### IMPORTANT
Our program might take a while to load during the very first time you run the program. When you first run, the model hasn't been trained yet. So it will take a while for the model to train. Subsequently after you run the program for the first time, it should take faster to run as the model has already been trained.

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
In our program there's a specific segment which reuqires customistation on your end. As the path to our datasets will be different from yours, it will require some changing.<br />
On line 15, it says:<br />`base_dir = '/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/Dataset.csv'`<br />You should replace it with:<br />`base_dir = '/Users/{yourusername}/path/to/fish-disease-diagnosis-py/Dataset.csv'`<br />^ Change it to the path in which our program is found on your computer.

## Step by Step Guide:
### Step 1:
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/6b2d9bda-56c4-4299-98b3-bb7ec4cd5576)




