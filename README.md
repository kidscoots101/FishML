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
Whenever the code requires for you to specify a path, for example:<br />`base_dir = '/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/Dataset.csv'` OR `healthy_img = preprocess_image("/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/healthy.png")`<br />You should replace it with:<br />`base_dir = '/Users/{yourusername}/path/to/fish-disease-diagnosis-py/Dataset.csv'`<br /> OR `healthy_img = preprocess_image("/Users/{yourusername}/path/to/fish-disease-diagnosis-py/healthy.png")`<br />^ Change it to the path in which our program is found on your computer. [NOTE: These are only examples. Please look through the entire program to see which part requires you to specify your own path.

## Step by Step Guide:
### Step 1:
Open our program (with datasets and main.py file) in your desired code editor.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/6b2d9bda-56c4-4299-98b3-bb7ec4cd5576)

### Step 2:
Open Main.py
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/33453b80-0367-4896-ada4-5d03af46c310)

### Step 3:
In your terminal go to the directory and type `streamlit run main.py`. Click enter.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/6427b73e-4831-4196-96d2-52531e379b58)

### Step 4:
You should see a screen pop up. If you don't, manually click on the links displayed in the terminal after completing Step 3.
![image](https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/edf88c3a-32b7-4a6b-a5f3-add77b76f61e)

### Step 5:
After the program opens, you should see something like this:
<img width="1440" alt="Screenshot 2024-02-15 at 7 10 40 PM" src="https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/f3d9a71c-a2a0-421f-aab2-40cf25c16d8b">

### Step 6:
Click on "Browse Files" and your finder should pop up as shown:
<img width="1440" alt="Screenshot 2024-02-15 at 7 11 32 PM" src="https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/260ae5c1-6fed-4f6d-ae1e-df41716f0ba7">

### Step 7:
Go to the `Dataset.csv/validation_set` and choose an image from the "healthy" or "diseased" class. The reason is because we trained our model on a different set of images, and by testing the model on a set of images different from the training images, it helps to prove the accuracy of our model.
<img width="1440" alt="Screenshot 2024-02-15 at 7 16 46 PM" src="https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/fd08dc3c-ff15-4bb5-9bcf-83f38089ae19">

### Step 8 (last step):
You should be directed back to the Home screen. The results of the prediction (either diseased or not diseased) is in the red box as shown in the image. The green box represnets a threshold. The threshold is the decision boundary seperating different classes. For example the threshold in this case is 0.11576238, and if the image exceeds this threshold, it's classified as "not diseased". Else if it doesn't exceed this threshold, it's calssifed as "diseased".
<img width="1440" alt="Screenshot 2024-02-15 at 7 18 12 PM" src="https://github.com/kidscoots101/fish-disease-diagnosis-py/assets/102847271/7364d50a-fd44-4e6d-9c1d-a3c62c6f3013">


## Thank you! ❤️








