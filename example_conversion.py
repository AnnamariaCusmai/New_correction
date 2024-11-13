import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

# Step 1: Define and Create a Simple PyTorch Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # Simple fully connected layer

    def forward(self, x):
        return self.fc(x)

# Create a sample PyTorch model and set it to evaluation mode
pytorch_model = SimpleModel()
pytorch_model.eval()

# Step 2: Create a Dummy Input for Exporting
# Note: The dummy input shape must match the model's expected input shape
dummy_input = torch.randn(1, 10)  # Batch size of 1 and 10 input features

# Step 3: Export the PyTorch Model to ONNX Format
onnx_filename = "simple_model.onnx"
torch.onnx.export(
    pytorch_model,            # The model being converted
    dummy_input,              # The dummy input for shape reference
    onnx_filename,            # The output ONNX file name
    input_names=['input'],    # The model's input name
    output_names=['output'],  # The model's output name
    opset_version=11          # ONNX opset version (adjust as necessary)
)

print(f"ONNX model exported to {onnx_filename}")

# Step 4: Load the ONNX Model
onnx_model = onnx.load(onnx_filename)

# Step 5: Convert the ONNX Model to TensorFlow
# Use the onnx-tf library's prepare method to convert
tf_rep = prepare(onnx_model)

# Step 6: Export the TensorFlow Model as a SavedModel
saved_model_dir = "tf_model"
tf_rep.export_graph(saved_model_dir)
print(f"TensorFlow model saved in directory: {saved_model_dir}")

# Step 7: Load the SavedModel and Save it as a .pb File
# The TensorFlow SavedModel format already contains the .pb file
# But we can further refine and control the export by loading and saving again

# Load the TensorFlow SavedModel
model = tf.saved_model.load(saved_model_dir)

# Define the export path for the `.pb` file
export_path = "converted_model"
if not os.path.exists(export_path):
    os.makedirs(export_path)

# Save the model to the `.pb` file
tf.saved_model.save(model, export_path)

print(f"Converted TensorFlow model saved to: {export_path}/saved_model.pb")

