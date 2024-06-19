# Facial Expression Recognition

This project aims to classify emotions from facial expressions using Keras and OpenCV in Python. The target emotions to be classified are happiness, sadness, anger, surprise, disgust, fear, and neutral.

## Dataset

The project uses the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) downloaded from Kaggle.  

## Methodology

## Libraries

The libraries used for the project are as follows:

* OpenCV (computer vision)
* NumPy (numerical operations)
* TensorFlow and Keras (deep learning)
* dlib (optional: facial landmark detection)

## Deliverables

## Getting Started

### Setting up your virtual environment:  

1. Create your virtual environment using the command:  
```powershell
python -m venv venv
```
* Notes: You are not limited to using venv as the name for you virtual environment. Don't forget to include the directory in your .gitignore when using git.

2. In the project directory, run the command:  
```powershell
.\venv\Scripts\activate
```
* When using powershell and an error appears about scripts being disabled on your system, run either of the following commands instead:  
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate
Set-ExecutionPolicy Default -Scope Process
```
or  
```powershell
get-ExecutionPolicy
Set-ExecutionPolicy Unrestricted -Force
.\venv\Scripts\activate
Set-ExecutionPolicy <get-ExecutionPolicy result> -Force
```

3. To ensure your virtual environment is activated, you may do the following:
    * Check for the `(venv)` prefix before your directory in the terminal
    * Check if there is a python path in your virtual environment directory in the terminal using `where python` or `where.exe python` in powershell

4. With your virtual environment activated, install the required dependencies using the following command in your terminal:

```powershell
pip install -r requirements.txt
```

5. When generating your requirements.txt, simply copy the results of the command below:
```powershell
pip freeze

<# Sample Results with pytz and requests packages installed

certifi==2024.6.2
charset-normalizer==3.3.2
idna==3.7
pytz==2024.1
requests==2.32.3
urllib3==2.2.1

#>
```

6. When you are done using the virtual environment, deactivate it using the following command in your terminal:  
```powershell
deactivate
```

## Contributing

## License

## References

This section lists resources that were helpful in the development of this project or provide further reading on the topic.

* Data Magic (by Sunny Kusawa). "Emotion Detection using CNN | Emotion Detection Deep Learning project |Machine Learning | Data Magic." YouTube, http://www.youtube.com/watch?v=UHdrxHPRBng.
* freeCodeCamp.org. "Deep Learning for Computer Vision with Python and TensorFlow – Complete Course." YouTube, https://www.youtube.com/watch?v=IA3WxTTPXqQ.
* Nicholas Renotte. "Build a Deep Face Detection Model with Python and TensorFlow | Full Course." YouTube, https://www.youtube.com/watch?v=N_W4EYtsa10.
* "Emotion Detection using Convolutional Neural Networks (CNNs)." GeeksforGeeks, https://www.geeksforgeeks.org/emotion-detection-using-convolutional-neural-networks-cnns/.
* "Adding a Rescaling Layer (or any Layer for that matter) to a Trained TensorFlow Model." Stack Overflow, https://stackoverflow.com/questions/66214588/adding-a-rescaling-layer-or-any-layer-for-that-matter-to-a-trained-tensorflow.
* [Keras API Documentation](https://keras.io/api/)
* [Tensorflow API Documentantation](https://www.tensorflow.org/api_docs/python/tf)