# Architecture Diagram Prompt for Image Generation

## Prompt for AI Image Generators (Imagen, DALL-E, Midjourney, etc.)

Create a professional neural network architecture diagram in the style of research papers (similar to ResNet, Transformer papers). The diagram should show:

**Layout:** Horizontal flow from left to right, with a central backbone that splits into three parallel branches.

**Left Side - Input:**
- A rectangle labeled "Input Image (112×112×3)" in light blue

**Center-Left - Backbone:**
- A large block labeled "IR-50 Backbone (ResNet-50 Variant)"
- Inside shows: "Input Layer → Body (Blocks 1-4) → Feature Maps (7×7×512)"
- **CRITICAL**: The body processes all 4 blocks sequentially and outputs 7×7×512 feature maps
- **NO split**: There is NO body_early/body_late - it's a single body that outputs 7×7×512
- All three branches (AdaFace, QAConv, OcclusionHead) branch from the SAME 7×7×512 feature maps
- Color: Beige/tan background

**Center - Three Branches (arranged vertically):**

**Top Branch (AdaFace - Blue tones):**
- Arrow from "Feature Maps (7×7×512)" → "Output Layer (Global Pooling)"
- Arrow → "AdaFace Embedding (512×1)"
- Arrow → "AdaFace Loss (Adaptive Margin Softmax)"
- Label: "Global Feature-Based Recognition"

**Middle Branch (QAConv - Purple tones):**
- Arrow from "Feature Maps (7×7×512)" → "Feature Normalization" (L2 normalization, NO downsampling)
- Arrow → "QAConv Features (7×7×512)"
- Separate arrow from "Feature Maps (7×7×512)" → "OcclusionHead" (showing 2 conv layers: Conv→BN→ReLU→Conv→Sigmoid)
- Arrow from "OcclusionHead" → "Occlusion Map (7×7×1)"
- Two arrows merge: "QAConv Features" + "Occlusion Map" → "QAConv Matcher"
- Inside "QAConv Matcher" box:
  - "Spatial Correlation (7×7 × 7×7)"
  - "Occlusion Weighting (element-wise)"
  - "Max Pooling → BatchNorm → FC"
- Arrow → "QAConv Similarity Score"
- Score splits into:
  - "Pairwise Loss (BCE)" with weight λ=1.0
  - "Triplet Loss (Margin)" with weight λ=0.5
- Both merge → "QAConv Loss"
- Label: "Local Feature Matching with Occlusion Awareness"

**Bottom Branch (Occlusion - Pink tones):**
- Arrow from "Feature Maps (7×7×512)" → "OcclusionHead"
- Detailed OcclusionHead structure:
  - "Conv2d(512→128, 3×3, padding=1)" (input: 7×7×512, output: 7×7×128)
  - "BatchNorm2d(128)"
  - "ReLU"
  - "Conv2d(128→1, 1×1)" (input: 7×7×128, output: 7×7×1)
  - "Sigmoid"
- Arrow → "Occlusion Map (7×7×1, 1=visible, 0=occluded)"
- Arrow → "Occlusion Loss (MSE with GT Masks)"
- Label: "Spatial Occlusion Prediction"

**Right Side - Loss Combination:**
- Large box labeled "Total Loss"
- Formula displayed:
  ```
  L_total = 0.1 × L_AdaFace 
          + 0.9 × L_QAConv 
          + 0.2 × L_Occlusion
  ```
- Color: Light red/pink

**Visual Style:**
- Clean, professional research paper aesthetic
- Boxes with rounded corners
- Arrows with clear direction
- Text labels at each stage
- Dimension annotations (e.g., "7×7×512")
- Color coding: Blue (AdaFace), Purple (QAConv), Pink (Occlusion)
- White background with subtle grid lines
- Minimalist, technical illustration style

**Key Annotations:**
- "Occlusion-aware local feature matching"
- "Adaptive margin based on image quality"

**Dimensions to show (ALL 7×7 for 112×112 input):**
- Input: 112×112×3
- Backbone Body output: 7×7×512 (after all 4 blocks, NO intermediate 14×14)
- Feature Maps (shared by all branches): 7×7×512
- QAConv Features: 7×7×512 (normalized, same spatial size)
- Occlusion Map: 7×7×1 (output from OcclusionHead)
- AdaFace Embedding: 512×1 (after global pooling)
- QAConv correlation: 49×49 (7×7 × 7×7 spatial correlation matrix)

**Additional Elements:**
- Small legend showing: "1 = visible, 0 = occluded" for occlusion maps
- Loss weight annotations (λada=0.1, λqa=0.9, λocc=0.2, λtri=0.5)
- Flow indicators showing forward pass direction
