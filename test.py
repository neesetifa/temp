LLM inference and serving

★ Multi-model routing / cascades (choosing a model under a cost budget)
★ Sampled-answer aggregation (best-of-N, majority voting, weighted voting)
★ Test-time compute allocation (how many samples per problem, when to stop)
Early exit / layer skipping policies
Semantic cache / prefix cache admission and eviction
Output length prediction
Constrained decoding / structured output (JSON, grammar constraints)
Batch packing and padding optimization for inference
Serving-side scheduling and swapping of LoRA adapters

Model compression

★ Low-bit compression of retrieval embeddings
Structured pruning / low-rank factorization
Weight quantization (mixed precision, codebook quantization)
Activation outlier handling (rotation, smoothing)
Vocabulary pruning / embedding table compression
Data and loss design for knowledge distillation

Training and optimization

★ Learning-curve extrapolation / early stopping
★ Scaling-law extrapolation
Learning-rate schedule design
Data mixture optimization
Curriculum learning / sample ordering
Gradient compression (distributed communication)
Loss-scaling strategy in mixed-precision training
Checkpoint averaging / EMA strategy

Data-centric

★ Weak-supervision label aggregation
★ Noisy-label detection and cleaning
Active-learning acquisition strategy
Data deduplication (near-duplicate detection)
Benchmark contamination detection
Synthetic-data filtering (quality scoring of LLM-generated data)
Data-augmentation policy search

Evaluation and reliability

★ Selective answering / abstention for LLMs
Hallucination detection (from logprobs or sample consistency)
Uncertainty calibration (classifiers, OOD)
OOD detection
Distribution-shift detection
Statistical tests for model comparison / confidence intervals for small-sample evaluation

Representation and interpretability

★ Predicting model performance from weights (weight-space learning)
Concept erasure (the A3 direction)
Sparse dictionary / SAE feature recovery (the A1 direction)
Probing design
Representation alignment between models (stitching, CKA)
Steering-vector construction

Retrieval and RAG

Reranking strategy for retrieval results
Document chunking strategy
Query rewriting / expansion
Hybrid retrieval score fusion (BM25 + dense)
Retrieval corpus pruning (which documents to keep in a fixed-size index)

Systems and compilers

Cross-hardware transfer of compiler cost models (the C1 direction)
Operator fusion / schedule search
Memory planning (which layers get activation checkpointing)
Partitioning strategy for pipeline parallelism
Tensor layout / memory allocation

RL and agents

Offline RL (learning a policy from logs)
Reward-model learning from noisy preferences
Preference-data selection (which pairs to use for DPO)
Tool-call planning and budget allocation for agents
