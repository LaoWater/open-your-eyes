# How LLMs Actually Process Images: The Architecture Behind Multimodal Intelligence

## The Fundamental Question

**No, it's not the same neural networks.** LLMs don't magically develop vision capabilities. Instead, multimodal LLMs use a sophisticated architecture that combines specialized vision components with the language model—but the key insight is that **the language model does the "thinking" once visual information is translated into its native format: tokens.**

## The Core Architecture: Vision-to-Language Bridge

### Step-by-Step Process

```
Image → Vision Encoder → Multimodal Connector → Language Model → Text Output
  ↓           ↓                    ↓                   ↓            ↓
Raw Pixels → Visual Features → Visual Tokens → Text Processing → Response
```

**1. Vision Encoder (Separate Neural Network)**
- **Purpose**: Extract meaningful features from images
- **Common Models**: ViT (Vision Transformer), CLIP vision encoder, SigLIP
- **Function**: Converts raw pixels into high-level visual representations
- **Example**: SigLIP-SO400M vision model, which employs a Shape-Optimized vision transformer (ViT) architecture with 400 million parameters

**2. Multimodal Connector/Projection Layer**
- **Purpose**: Translate visual features into language model's "vocabulary"
- **Function**: Maps visual embeddings to text embedding space
- **Key Insight**: The extracted visual features are mapped (and optionally pooled) to the language model (LLM) input space, creating visual tokens

**3. Language Model Processing**
- **Purpose**: Process the combined visual + text tokens
- **Reality**: These visual tokens are concatenated (and potentially interleaved) with the input sequence of text embeddings
- **Result**: The LLM processes this as if it were reading a very long text with special "visual words"

## Two Main Architectural Approaches

### Method A: Unified Embedding (Decoder-Only)

```python
# Conceptual architecture
class UnifiedMultimodalLLM:
    def __init__(self):
        self.vision_encoder = ViT()
        self.projection_layer = LinearProjection()
        self.llm = DecoderOnlyLLM()  # Same as text-only LLM
    
    def process(self, image, text):
        # Extract visual features
        visual_features = self.vision_encoder(image)
        
        # Project to LLM token space
        visual_tokens = self.projection_layer(visual_features)
        
        # Tokenize text
        text_tokens = self.llm.tokenizer(text)
        
        # Concatenate and process as unified sequence
        combined_tokens = torch.cat([visual_tokens, text_tokens])
        response = self.llm.generate(combined_tokens)
        
        return response
```

**How it works**: The concept behind the hybrid model (NVLM-H) is to combine the strengths of both methods: an image thumbnail is provided as input, followed by a dynamic number of patches passed through cross-attention to capture finer details.

### Method B: Cross-Attention Based

```python
class CrossAttentionMultimodalLLM:
    def __init__(self):
        self.vision_encoder = ViT()
        self.llm_with_cross_attention = ModifiedLLM()  # Added cross-attention layers
    
    def process(self, image, text):
        # Visual features stay separate
        visual_features = self.vision_encoder(image)
        text_tokens = self.tokenize(text)
        
        # LLM attends to visual features via cross-attention
        response = self.llm_with_cross_attention.generate(
            text_tokens, 
            visual_context=visual_features
        )
        return response
```

**Trade-offs**: As the cross-attention layers add a substantial amount of parameters, they are only added in every fourth transformer block to balance performance and efficiency.

## The "Token Translation" Magic

**Key Insight**: LLMs can only process sequences of tokens. The breakthrough is making images "speak the language" of tokens.

### Visual Tokenization Process

1. **Image Patches**: The image encoder takes 224×224 resolution images and divides them into a 14×14 grid of patches, with each patch sized at 16×16 pixels

2. **Feature Extraction**: Each patch becomes a high-dimensional vector

3. **Projection to Text Space**: Visual vectors are mapped to the same dimensional space as text embeddings

4. **Token Sequence**: The result is a sequence of "visual tokens" that the LLM treats like special words

### Example: How GPT-4V "Sees" an Image

```
Original Image: [Kitchen scene with person cutting vegetables]

After Vision Processing:
Visual Token 1: [0.23, -0.45, 0.67, ...] → Represents "person's hand"
Visual Token 2: [0.12, 0.89, -0.34, ...] → Represents "knife blade"  
Visual Token 3: [-0.56, 0.78, 0.23, ...] → Represents "chopping board"
...
Visual Token N: [0.34, -0.67, 0.45, ...] → Represents "vegetable pieces"

Text Tokens: ["What", "safety", "concerns", "do", "you", "see", "?"]

Combined Input to LLM:
[VIS_TOK_1, VIS_TOK_2, ..., VIS_TOK_N, "What", "safety", "concerns", "do", "you", "see", "?"]
```

The LLM processes this exactly like it would process text, but some of the "words" happen to represent visual concepts.

## Training Process: How They Learn to "See"

### Phase 1: Vision-Language Alignment
- **Objective**: Teach the projection layer to map visual features to meaningful text space
- **Method**: The pre-trained model was first primed to predict the next word, using a large dataset of text and image data from the Internet and licensed data sources
- **Result**: Visual tokens gain semantic meaning in language space

### Phase 2: Instruction Tuning
- **Objective**: Teach the model to follow visual instructions
- **Method**: By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding

## Why This Architecture Works

### 1. Leveraging Pre-trained Strengths
- **Vision Encoder**: Already trained on massive image datasets
- **Language Model**: Already trained on massive text datasets  
- **Combination**: Gets the best of both without starting from scratch

### 2. Token-Based Unification
The genius is that **everything becomes tokens**:
- Text → Text tokens
- Images → Visual tokens  
- Audio → Audio tokens (in models like GPT-4o)

The LLM just sees a sequence of tokens and doesn't "know" which ones came from images vs. text.

### 3. Emergent Reasoning
The surprising emergent capabilities of the MLLM, such as writing stories based on images and optical character recognition–free math reasoning, are rare in traditional multimodal methods

## Practical Implications for Your Platform

### Computational Reality
- **Vision Encoder**: Runs once per image (moderate cost)
- **Projection Layer**: Very lightweight (minimal cost)
- **LLM Processing**: Scales with response length (major cost)

### Deployment Considerations
1. **Vision Encoder can be optimized**: Llama 3-V combines the Llama3 8B language model with the SigLIP-SO400M vision model - much smaller than SAM
2. **Edge deployment possible**: The 8B model outperforms GPT-4V, Gemini Pro, and Claude 3 across 11 public benchmarks, processes high-resolution images at any aspect ratio
3. **Efficient processing**: Single forward pass vs. frame-by-frame analysis

## Bottom Line

**LLMs don't inherently "see" images.** They process visual information that has been translated into their native language: tokens. The architecture is:

1. **Specialized vision component** extracts visual features
2. **Translation layer** converts features to text-like tokens  
3. **Same LLM architecture** processes visual + text tokens together

This is why multimodal LLMs can be so adaptable—they're using the same powerful reasoning capabilities that work for text, just fed with visual information that's been translated into "visual vocabulary."

For your video intelligence platform, this suggests that **hybrid approaches** (fast CV + LLM reasoning) might be the sweet spot: use specialized computer vision for efficient detection, then leverage LLM multimodal capabilities for complex reasoning about what those detections mean.