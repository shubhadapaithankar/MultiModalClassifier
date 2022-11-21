# MultiModalClassifier
This is a project repo for multi-modal deep learning classifier with popular models from Tensorflow and Pytorch. The goal of these baseline models is to provide a template to build on and can be a starting point for any new ideas, applications. If you want to learn basics of ML and DL, please refer this repo: https://github.com/lkk688/DeepDataMiningLearning.

# Package setup
Install this project in development mode
```bash
(venv38) MyRepo/MultiModalClassifier$ python setup.py develop
```
After the installation, the package "MultimodalClassifier==0.0.1" is installed in your virtual environment. You can check the import
```bash
>>> import TFClassifier
>>> import TFClassifier.Datasetutil
>>> import TFClassifier.Datasetutil.Visutil
```

If you went to uninstall the package, perform the following step
```bash
(venv38) lkk@cmpeengr276-All-Series:~/Developer/MyRepo/MultiModalClassifier$ python setup.py develop --uninstall
```

# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# Modified Changes as fallows

## Colab Link : https://colab.research.google.com/drive/1Vrx3T7wf8gExZHyDh82pIEWGPHOWRpjc#scrollTo=WplVQH5ilkMK 
## code modification Commits : https://github.com/lkk688/MultiModalClassifier/commit/985da0181c4a9d4c4a4e1a8789773a2c7857d851 

Steps : 
DATASET - 
1. Downloaded the dataset of dog and cat consisting of 4000 samples.
2. Tensorflow framework has been used to train the dataset.
3. Loaded the dataset using the datasetutil file i.e TFdatsetutil.py.
4. Trained the model on google colab using the python script i.e myTFDistributedTrainerv2.py
  `  !python3 -m TFClassifier/myTFDistributedTrainerv2 --data_name custom --data_path "/content/drive/MyDrive/Shubhada/MultiModalClassifier/cats_and_dogs_filtered/train/" --img_height 200 --img_width 200 --save_path "/content/drive/MyDrive/Shubhada/MultiModalClassifier/output" --batchsize 32 `
    
5. Trained the model for 50 epochs and got the loss function i.e 
 `50/50 [==============================] - 3s 52ms/step - loss: 0.4410 - accuracy: 0.8125 - val_loss: 0.5859 - val_accuracy: 0.7000 - lr: 1.2097e-05
 FINAL ACCURACY MEAN-5:  0.6579999923706055
 TRAINING TIME:  54.18541431427002  sec `
 
6. Took the inference for val dataset using i.e myTFInference.py
 ` !python TFClassifier/myTFInference.py --data_name custom --data_path "/content/drive/MyDrive/Shubhada/MultiModalClassifier/cats_and_dogs_filtered/validation/" --img_height 200 --img_width 200 --model_path /content/drive/MyDrive/Shubhada/MultiModalClassifier/outputcustom_cnnsimple1_0629`
  
7. Got some good prediction on val data with a confidence of 94.
8. The trained weights was used to convert .pb files to tflite to get the inference.
9. Quantization of unint8 to int8 conversion was done.
10. Model size was drastically reduced from 80 MB to 20 MB for faster predictions.
11. The .pb files were converted to onnx format using tf2onnx package.
12. Results:
  ` !python3 -m onnxsim '/content/drive/MyDrive/Shubhada/MultiModalClassifier/onnx/custom.onnx' "/content/drive/MyDrive/Shubhada/MultiModalClassifier/onnx/custom_sim.onnx" --input-shape    1,200,200,3`
13. ONNX Models can be used to convert to any format.
14. Conversion of models using openvino to run the inference
 ` !mo --saved_model_dir /content/drive/MyDrive/Shubhada/MultiModalClassifier/outputcustom_cnnsimple1_0629/ --input_shape "[1,200,200,3]`
  
15. CONVERSION OF .PB files to .xml and .bin files (i.e OPEN VINO RUNTIME)
16. Using the openvino format , i achieved a inference time of i.e
   `  IR model in OpenVINO Runtime/CPU: 0.0086 seconds per image, FPS: 116.80`
17. Tensorflow model vs Open Vino
18. Tensorflow - Inference - 0.5 fps | modelsize= 80 MB
19. Tensorflowlite - Inference - 5 fps | modelsize = 20 MB
20. Openvino   - Inference - 116 FPS | modelsize = 22 KB


Tensorflow flow model was pretty slow , i.e 5 fps or less in terms of giving the prediction.
Further i converted the trained model to onnx format where i further got it converted to openvino runtime.
From the above results, it pretty much gave  me 116 fps with good prediction as well.
