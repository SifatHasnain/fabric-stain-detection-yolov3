# FABRIC STAIN DETECTION USING YOLOv3

This implementation is based on [this](https://github.com/PrimeshShamilka/fabric_defect_detector) project and integrates a REST API to it for detecting stain defects from an image.


### How to run?

#### For Linux 

- Run the following commands in a terminal

```
   git clone https://github.com/SifatHasnain/fabric-stain-detection-yolov3.git
   cd fabric-stain-detection-yolov3
   python -m venv venv
   source venv/bin/activate
   make
   pip install -r requirements.txt
```

- Donwload the configuration file of the model from the link below and copy it to cfg directory inside darknet root directory \
  https://drive.google.com/file/d/12ikV938ZEXWjoITYW6mE0FqoZQbyET1v/view?usp=sharing
  
- Download obj.data, classes.names files from the links below and copy them to data directory inside darknet root directory
  obj.data file - https://drive.google.com/file/d/1B54Q4VQjlHVkLoQb132EgBjyEJ0lJ0yy/view?usp=sharing \
  classes.names file - https://drive.google.com/file/d/1WBVQI6e0p7TQtTExXbRUcwaG5N9A9xwL/view?usp=sharing

- Download the trained weight file for the model from the link below and copy the weights to backup directory inside darknet root directory \
  https://drive.google.com/file/d/18GULkIGA2Qpg4rfYbp2moGaZLoxtjuAw/view?usp=sharing

- Run the server

``` 
  python app.py
```

- API Endpoint http://127.0.0.1:5000/detection/stain. You can test the endpoint with data in the following json format

```
{
  'data': base64_image_string
}
```

### Train 

https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

## REFERENCES

- https://github.com/PrimeshShamilka/fabric_defect_detector

- https://github.com/AlexeyAB/darknet

