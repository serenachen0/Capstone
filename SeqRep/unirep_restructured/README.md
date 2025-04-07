# UniRep Refactoring & Migration Plan

## Phase 1: Assessment and Infrastructure Setup (March 3 - March 8) :white_check_mark:

1. Create development environment with TensorFlow 2.18 :white_check_mark:
2. Create version control branch for refactoring :white_check_mark:
3. Run original model with sample inputs and save outputs for comparison :white_check_mark:
4. Profile current model to identify specific performance bottlenecks :white_check_mark:
5. Create UML Diagram for existing class structure :white_check_mark:
6. Propose UML Diagram of re-factored class structure :white_check_mark:

## Phase 2: Models Restructuring (March 9 - March 15)

1. Prepare and Design Project File Structure :white_check_mark:
2. Design class hierarchy based on the updated model diagram: :white_check_mark:
   - Configuration System: Create `BaseConfig` and `UniRepModelConfig` classes :white_check_mark:
   - Implement proper inheritance for LSTM cells :white_check_mark:
   - Design separation of concerns between model components :white_check_mark:
3. Refactor existing code into proper classes: :white_check_mark:
   - Extract `SequenceProcessor` from utility functions :white_check_mark:
    - Create dedicated `RepresentationUtils` class :white_check_mark:
   - Implement `BaseLSTMCell` and `mLSTMCell` classes :white_check_mark:
4. Add proper encapsulation of model components: :white_check_mark:
   - Implement proper getter/setter methods :white_check_mark:
   - Create consistent initialization patterns :white_check_mark:
   - Ensure clear interface boundaries between components :white_check_mark:
5. Create unit tests for each class component :white_check_mark:
6. Validate that refactored structure produces equivalent outputs 

## Phase 3: TensorFlow 2.0 Migration (March 16 - March 22)

1. Migrate each refactored class to TensorFlow 2.0: :white_check_mark:
   - Convert TF 1.3 ops to TF 2.0 equivalents :white_check_mark:
   - Replace sessions with eager execution or `tf.function` :white_check_mark:
   - Update variable management :white_check_mark: :white_check_mark:
2. Implement Keras-based model structure: :white_check_mark: 
   - Convert `mLSTMCell` to Keras layer :white_check_mark:
   - Create model class using Keras subclassing API :white_check_mark:
   - Implement equivalent initialization approaches :white_check_mark:
3. Convert placeholders to TensorFlow 2.0 input methods :white_check_mark:
4. Update model loading/saving to SavedModel format 
5. Ensure backward compatibility with existing model weights

## Phase 4: Fix Non-Deterministic Behavior in TF 2.0 (2 weeks)

1. Implement global seed control with TF 2.0 APIs :white_check_mark:
2. Configure TensorFlow 2.0 for deterministic operations `tf.config.experimental.enable_op_determinism()`
3. Replace any remaining non-deterministic operations 
4. Implement deterministic versions of critical functions
5. Add validation tests for deterministic behavior
6. Document configuration for reproducibility 

## Phase 5: Improve Performance in TensorFlow 2.0 (March 23 - March 29)

1. Implement efficient batching with TF 2.0 data pipelines :white_check_mark:
2. Optimize for GPU processing: 
   - Implement memory-efficient operations
   - Explore other GPU processing optimization techniques: Floating Point Mixing
3. Add `BatchProcessor` class for parallel execution
4. Optimize model architecture for better throughput
5. Implement TF 2.0-specific performance enhancements
6. Benchmark and document performance improvements

## Phase 6: Testing, Documentation and Delivery (1-2 weeks)

1. Compare outputs with original TF 1.3 models
2. Verify deterministic behavior across multiple runs
3. Test with varying batch sizes and sequence lengths
4. Measure and document performance improvements
5. Create detailed documentation for the new implementation
6. Document configuration options for determinism
7. Provide usage examples for protein sequence analysis
8. Update UML diagrams of the new object-oriented structure
9. Document performance best practices and benchmarks
10. Prepare final report comparing with original implementation







## Phase 1:  & Base Components

1. **Setup Project Structure**
   - Create directories as per proposed file structure
   - Set up package files and imports
   - Create setup.py for installation
2. **Configuration System**
   - Implement BaseConfig with validation logic
   - Implement UniRepModelConfig with model-specific parameters
   - Add methods for loading/saving configurations
3. **Utilities Implementation**
   - Implement sequence processing utilities (encoding/decoding)
   - Implement tensor utilities (including deterministic operation handling)
   - Implement representation utilities

## Phase 2: Core Components Implementation

1. **Base LSTM Cell**
   - Implement BaseLSTMCell with TensorFlow 2.x compatibility
   - Add deterministic operation support
   - Include proper GPU handling
2. **mLSTM Cell Implementation**
   - Refactor mLSTMCell using BaseLSTMCell
   - Ensure weight initialization is deterministic
   - Add GPU optimization options
3. **mLSTM Stacked Cell**
   - Implement stacked cell architecture
   - Support residual connections
   - Support dropout (for training purposes)
4. **Base Model Implementation**
   - Set up abstract base model with common functionality
   - Implement embedding layer
   - Set up placeholder and graph management
   - Support both eager and graph execution

## Phase 3: Model Implementations

1. **UniRep64 Implementation**
   - Implement 64-unit model
   - Set up weight loading/initialization
   - Ensure TensorFlow 2.x compatibility
2. **UniRep256 Implementation**
   - Implement 256-unit model
   - Set up weight loading/initialization
   - Ensure TensorFlow 2.x compatibility
3. **UniRep1900 Implementation**
   - Implement 1900-unit model
   - Set up weight loading/initialization
   - Ensure TensorFlow 2.x compatibility

## Phase 4: CLI & Integration

1. **Command Line Interface**
   - Implement seq2rep.py with modern argument parsing
   - Implement rep_compare.py for representation comparison
   - Add proper logging and progress feedback
2. **Batch Processing Optimization**
   - Implement efficient batch processing for sequences
   - Add parallel processing support
   - Optimize memory usage

## Phase 5: Testing & Benchmarking

1. **Unit Tests**
   - Set up test framework
   - Implement tests for each component
   - Ensure reproducibility tests
2. **Integration Tests**
   - Test end-to-end workflows
   - Test different model sizes
   - Test with representative protein sequences
3. **Benchmarking**
   - Compare performance with original implementation
   - Measure GPU utilization
   - Test scalability with different batch sizes

## Phase 6: Documentation & Finalization

1. **Code Documentation**
   - Add docstrings to all functions and classes
   - Generate API reference documentation
2. **Usage Examples**
   - Create Jupyter notebooks with examples
   - Document common use cases
3. **Final Refactoring**
   - Address any technical debt
   - Optimize critical code paths
   - Final code review and cleanup
