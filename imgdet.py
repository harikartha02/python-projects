from imageai.Detection import ObjectDetection  
    
recognizer = ObjectDetection()  
  
path_model = "C:/Users/karth/OneDrive/Desktop/acm/yolo-tiny.h5"  
path_input = "C:/Users/karth/OneDrive/Pictures/Saved Pictures/new_img.jpg"  
path_output = "C:/Users/karth/OneDrive/Pictures/Saved Pictures/detected_img.jpg"  
  
recognizer.setModelTypeAsTinyYOLOv3()  
 
recognizer.setModelPath(path_model)  

recognizer.loadModel()  
  
recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  
    
for eachItem in recognition:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  

