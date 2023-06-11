# Gait-Analysis-
Gait analysis (GA) toolkit is a machine learning-based human activity recogntion developed for detecting common activities of daily living (ADLs), 
such as walking, jogging, going upstairs, going downstairs, sitting, and standing. The GA toolkit contains a pre-trained model based on the smartphone 
acceleration dataset obtained from wearable inertial sensors.

Installation Instructions

The GA toolkit has been packaged in a dockerized container. 

The toolkit’s image can be tested and run on a  local machine  and can be accessed by running the following commands : 

Run GA component: docker run -d -p 8082:8080 gait_module 

Currently, the GA image can be accessed and tested from Docker hub by typing the following commands: 

•	Step 1: type docker pull adanentnu/gait_module (on terminal) 

•	Step 2: type docker run -p 8088:8080 adanentnu/gait_module (on terminal) 

•	Step 3: type localhost:8088/apidocs ( on a browser)

