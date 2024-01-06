from superpoint import SuperPoint
import torch
import os
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    # TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER

# Conversion is done using: Pytorch model --> ONNX model --> TensorRT

def main():
    dir = os.path.dirname(__file__)

    # Load model
    model_weights_path = os.path.join(dir, "weights/superpoint_v1.pth")
    model = SuperPoint(SuperPoint.default_config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # set the model to inference mode
    model.eval()

    # Input is in NCHW format
    # N: Batch number
    # C: Channels, but grayscale only has 1
    # H: Height of image
    # W: Width of image
    batch_size = 1
    height = 240  # dynamic axes
    width = 320  # dynamic axes
    input = torch.randn(batch_size, 1, width, height)

    # onnx_filename = os.path.join(dir, weights.split("/")[-1].split(".")[0] + ".onnx")
    onnx_filename = model_weights_path.replace(".pth", ".onnx")

    # Export the model
    # export_options = torch.onnx.ExportOptions(dynamic_shapes=False)
    # export_output = torch.onnx.dynamo_export(model, kpts0, scores0, desc0, kpts1, scores1, desc1, export_options = export_options)
    # export_output.save(onnx_filename)

    torch.onnx.export(
        model,  # model being run
        input,  # model input (or a tuple for multiple inputs)
        onnx_filename,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model input names
        output_names=["scores", "descriptors"],  # the model output names
        dynamic_axes={
            "input": {0: "batch_size", 2: "image_height", 3: "image_width"},
        },  # dynamic model input names
    )

    # Tensorrt engine
    profiles = [
        Profile().add("input", min=(1, 1, 144, 144), opt=(1, 1, 240, 320), max=(1, 1, 1080, 1920)),
    ]
    engine = engine_from_network(
        network_from_onnx_path(onnx_filename), config=CreateConfig(profiles=profiles)
    )
    save_engine(engine, model_weights_path.replace(".pth", ".engine"))

if __name__ == "__main__":
    main()