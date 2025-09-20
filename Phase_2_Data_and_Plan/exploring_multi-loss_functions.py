# Example of Architecture

# Ground Truth: [Hand without gloves at (100,150)]
# Model Prediction: [Hand with gloves at (105,152)] ❌

# Standard Loss Calculation:
# - Objectness: ✅ Correctly found object (low loss)
# - Classification: ❌ Wrong class (medium loss) 
# - Localization: ✅ Close position (low loss)
# → Total Loss: 2.3

# Safety-Critical Loss Calculation:
# - Objectness: ✅ Correctly found object (low loss)
# - Classification: ❌ Wrong class × 4.0 weight = HIGH LOSS!
# - Localization: ✅ Close position (low loss)  
# → Total Loss: 7.8 (much higher!)