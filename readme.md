# Florence-2 C++ Port

Welcome to the C++ port of the Florence-2 AI model! 

This project allows you to run Microsoft's powerful Florence-2 vision-language AI completely offline, locally on your own machine. Instead of relying on massive cloud GPUs or Python servers, this project uses highly optimized C++ and the ONNX Runtime to crunch the AI's math directly on your CPU.

## 🧠 How The Magic Works

At its core, AI is just math. Because computers only understand numbers, the AI model cannot "see" a JPEG image or "read" an English sentence directly. Here is the step-by-step flow of how this C++ engine works:

```mermaid
graph TD
    %% Input Layer
    UserImage[📷 Image File] --> |Bytes| Florence2Model
    UserText[💬 Text Prompt] --> |String| Florence2Model
    
    %% Core Model Class
    subgraph Florence2Model [Florence2Model (The Main Brain)]
        
        %% Step 1: Pre-processing
        subgraph PreProcessing [1. Pre-Processing]
            Tokenizer[🔤 Tokenizer]
            ImageProcessor[🖼️ CLIP Image Processor]
        end
        
        %% Step 2: ONNX Math Engines
        subgraph ONNXEngines [2. ONNX Math Engines]
            VisionEncoder[👁️ Vision Encoder]
            TextEmbedder[📝 Text Embedder]
            MainEncoder[🧠 Main Encoder]
            Decoder[🗣️ Decoder]
        end
        
        %% Step 3: Post-processing
        subgraph PostProcessing [3. Post-Processing]
            OutputTokenizer[🔤 Tokenizer]
            PostProcessor[✨ Post Processor]
        end
        
        %% The Flow Connections
        Tokenizer --> |Numbers| TextEmbedder
        ImageProcessor --> |Pixels| VisionEncoder
        
        VisionEncoder --> |Image Features| MainEncoder
        TextEmbedder --> |Text Embeddings| MainEncoder
        
        MainEncoder --> |Hidden States| Decoder
        
        Decoder --> |Number IDs| OutputTokenizer
        OutputTokenizer --> |English Text| PostProcessor
    end

    %% Output Layer
    PostProcessor --> |Final Result| Result[Result: 'A cat on a couch']

    %% Styling
    style Florence2Model fill:#f9f9f9,stroke:#333,stroke-width:2px
    style PreProcessing fill:#e1f5fe,stroke:#03a9f4
    style ONNXEngines fill:#e8f5e9,stroke:#4caf50
    style PostProcessing fill:#fff3e0,stroke:#ff9800
    style Result fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
```

1. **Pre-Processing:** The `Tokenizer` turns your English prompt into a sequence of ID numbers. The `CLIPImageProcessor` chops your image up into pixel grids.
2. **ONNX Engines:** The raw numbers are fed into the massive ONNX mathematical models (`.onnx` files). The math is split into 4 parts: understanding the image, understanding the text, combining both understandings, and finally "decoding" the answer one number at a time.
3. **Post-Processing:** The output numbers are fed *back* into the Tokenizer to be translated back into readable English sentences!

---

## 🚀 How to Run It Yourself

Follow these steps to run the AI from scratch on your own machine.

### Step 1: Build the Project
First, compile the C++ code into executable binaries using CMake:
```bash
mkdir build
cd build
cmake ..
cmake --build . -j8
```

### Step 2: Download the AI Models
The C++ code is just the engine; it needs the "brain" (the multi-gigabyte `.onnx` model files) to actually work! We have provided a built-in downloader that will fetch the models from Hugging Face for you.

Run this command from the root of the project to download the models into the `./models` folder:
```bash
# On Mac, ensure ONNX Runtime is in your library path first
export DYLD_LIBRARY_PATH=/usr/local/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH

./build/bin/test_florence_model_downloader_implementation
```
*Wait until you see "All models download: OK". This may take a few minutes!*

### Step 3: Run the AI Inference
Now that the engine is compiled and the models are downloaded, you can run the AI!
```bash
export DYLD_LIBRARY_PATH=/usr/local/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH
./build/bin/test_florence2_inference
```

---

## 🖼️ How to Upload Your Own Image

Want to see what the AI thinks of your dog? It's incredibly easy to test your own images.

1. **Copy your image** into the `tests/test_image/` folder.
2. **Open** `tests/unit_tests/test_florence2_inference.cpp` in a text editor.
3. **Find** the `test_basic_inference()` function (around line 114).
4. **Change the path** to point to your new image:
   ```cpp
   // Change this line:
   auto image_data = load_test_image("./tests/test_image/image copy 3.png");
   
   // To this:
   auto image_data = load_test_image("./tests/test_image/my_dog.png");
   ```
5. **Recompile** the code (so it grabs your changes!):
   ```bash
   cmake --build build -j8
   ```
6. **Run it!**
   ```bash
   export DYLD_LIBRARY_PATH=/usr/local/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH
   ./build/bin/test_florence2_inference
   ```

You can also change the `TaskType::MORE_DETAILED_CAPTION` inside that file to other tasks, like `TaskType::CAPTION` or `TaskType::OCR` to ask the AI to read text from the image!