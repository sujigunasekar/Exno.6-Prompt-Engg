# Exno.6-Prompt-Engg
## Date:
## Register no: 212222230152
## Aim: 
Development of Python Code Compatible with Multiple AI Tools

## Algorithm: 
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.

## Objective:
Integrate image detection using the Google Vision API to analyze potential images of spy cameras.

Use OpenAI GPT-4 to interpret the analysis and describe potential threats or suspicious devices.

Compare the outputs of both tools and generate actionable insights.

# Procedure / Algorithm:
## Step 1: 
Define the Use Case
We are developing a system to detect spy cameras in various environments, such as hotel rooms or bathrooms. The tools will process image inputs and provide information about potential hidden devices, generating a summary of the findings.

Example Use Case: Detect hidden cameras in images (spy cams), analyze the results, and provide a descriptive output for actionable insights.

## Step 2:
Set Up API Access
You need API keys for the services you will be using:

Google Cloud Vision API for image analysis.

OpenAI API (GPT-4) for text-based interpretation and descriptive output.

Store API keys securely (e.g., using environment variables).
```
import os
```
## Fetch API keys from environment variables
```
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
## Step 3:
Write API Interaction Functions

## Google Vision API Function (Image Analysis):
This function will take an image as input and detect objects within it. If there are any suspicious devices like cameras, it will return related information.
```
from google.cloud import vision
from google.cloud.vision import types
```
## Function to detect objects in an image using Google Vision API
```
def detect_spy_cams(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = client.object_localization(image=image)
    
    objects = response.localized_object_annotations

    spy_cam_detected = []
    for object_ in objects:
        if 'camera' in object_.name.lower():
            spy_cam_detected.append(object_.name)

    return spy_cam_detected
```

## OpenAI GPT-4 Function (Text Interpretation):
After identifying potential spy cams in images, GPT-4 will provide an interpretation and generate a description of the findings.
```
import openai
```
## Function to interpret spy cam detection using OpenAI's GPT-4
```
def interpret_with_openai(detected_objects):
    openai.api_key = OPENAI_API_KEY
    
    # Create a prompt based on the detected objects
    prompt = f"The following objects have been detected in the image: {', '.join(detected_objects)}. Based on these findings, please provide a detailed description of potential spy cameras and their threat level."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['choices'][0]['message']['content']
```

## Step 4:
Prepare a Dataset
We can test the system with different images that could potentially contain hidden cameras.

## List of images to test (paths to image files)
```
test_images = [
    "test_image1.jpg",
    "test_image2.jpg",
    "test_image3.jpg"
]
```
## Step 5:
Compare Outputs Programmatically
For each test image, we'll:

Run the image through Google Vision API to detect potential spy cams.

If any objects are detected, we pass them to OpenAI GPT-4 to interpret and provide insights.
```
results = []

for image in test_images:
    # Detect spy cams in the image using Google Vision
    detected_objects = detect_spy_cams(image)
    
    if detected_objects:
        # If spy cams are detected, interpret the results using OpenAI GPT-4
        interpretation = interpret_with_openai(detected_objects)
        result = {
            "image": image,
            "detected_objects": detected_objects,
            "interpretation": interpretation
        }
        results.append(result)
```
## Step 6: 
Generate Actionable Insights
For each result, you can generate actionable insights like:

Number of cameras detected.

Threat level (e.g., high/medium/low).

## Suggest preventive measures or actions.
```
def generate_insights(result):
    # Example of generating actionable insights
    threat_level = "High" if "camera" in result['detected_objects'] else "Low"
    insights = {
        "image": result['image'],
        "threat_level": threat_level,
        "suggested_action": "Consider further inspection or reporting the issue." if threat_level == "High" else "No immediate threat detected."
    }
    return insights
```
## Apply the insight generation to each result
```
insights_results = [generate_insights(result) for result in results]
```
## Step 7: Output a Summary Report
Save the findings and insights to a structured file format (e.g., JSON or CSV) for further examination or reporting.
```
import json
```
## Save the comparison results and actionable insights as a JSON file
```
with open("spy_cam_detection_report.json", "w") as f:
    json.dump(insights_results, f, indent=4)
```
## Result:
The Python code successfully integrates Google Vision API for image-based detection of potential spy cameras and OpenAI GPT-4 for generating text-based interpretations of those findings. The system:

1.Detects suspicious devices in the image (such as hidden cameras).

2.Interprets the findings with actionable insights.

3.Generates a detailed report on the potential threats.

## Conclusion:
This approach demonstrates how Python can serve as a bridge between multiple AI tools for effective Spy Cam Detection. The integration allows for:

Image-based detection of hidden devices.

Text-based analysis and interpretation of detected objects.

Actionable insights to help detect, assess, and mitigate potential spy camera threats.

This multi-tool integration can be applied in various scenarios, including security systems, privacy protection, and surveillance.

This process helps in benchmarking AI tools and determining the best tool for a particular task or use case.

## Result: 
The corresponding Prompt is executed successfully
