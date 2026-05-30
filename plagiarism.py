"""
plagiarism.py  ── v5 (Master's thesis grade)
════════════════════════════════════════════
Changes from v4
───────────────
IMPROVEMENT 1 — Sigmoid calibration replaces linear z-score scaling
  Root cause of instability: linear scaling amplifies small std deviations
  in a homogeneous corpus, causing disproportionate score swings.

  Fix: Sigmoid normalisation centred on corpus mean:
    normalised = 1 / (1 + exp(-k * z))
  where k=3 controls steepness. This compresses extreme values, is
  numerically stable regardless of corpus_std magnitude, and produces
  a smooth monotonic mapping without hard clipping artifacts.

IMPROVEMENT 2 — Configurable thresholds (no longer hardcoded)
  Thresholds are dataset-dependent by nature. They are now exposed as
  constructor parameters with sensible defaults, allowing per-deployment
  calibration without code changes.

IMPROVEMENT 3 — Diversified corpus (200 sentences, 10 domains)
  The original 60-sentence AI/NLP corpus caused domain-bias: any on-topic
  text scored high regardless of actual copying. The expanded corpus adds:
    • General Science & Technology
    • Climate & Environment
    • Economics & Finance
    • Law & Ethics
    • History & Society
  This broadens the background distribution and reduces false positives
  for AI/ML domain text.

IMPROVEMENT 4 — Aggregation uses median alongside max/mean
  Formula revised to:
    agg = 0.50 × max_norm + 0.30 × mean_norm + 0.20 × median_norm
  Median is robust to a single outlier sentence inflating the aggregate.

No changes to FAISS index logic or public API signatures.
"""

import re
import faiss
import numpy as np
from model_loader import get_embedder


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS  (200 sentences, 10 topical domains)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CORPUS = [
    # ── AI / General (20) ─────────────────────────────────────────────────────
    "Artificial intelligence is increasingly used in modern systems to improve efficiency and automate complex tasks.",
    "AI-powered systems are transforming industries by automating repetitive tasks and enhancing productivity.",
    "Modern AI applications integrate machine learning and rule-based systems to solve real-world problems.",
    "Artificial intelligence refers to the simulation of human intelligence processes by computer systems.",
    "AI systems can learn from experience, adjust to new inputs, and perform human-like tasks.",
    "The rapid adoption of AI technologies is reshaping the global economy and labor markets.",
    "Intelligent systems use data-driven models to make predictions and support human decision-making.",
    "AI applications span healthcare, finance, education, transportation, and many other domains.",
    "Governments and organisations are developing frameworks to ensure the responsible use of AI.",
    "Ethical concerns around AI include bias, transparency, accountability, and privacy violations.",
    "Explainable AI aims to make the decision-making process of models interpretable to human users.",
    "AI safety research focuses on ensuring that intelligent systems behave in alignment with human values.",
    "The development of general artificial intelligence remains one of the most debated topics in computer science.",
    "Autonomous systems use AI to perceive their environment and take actions without human intervention.",
    "AI benchmarks such as ImageNet and GLUE are used to compare model performance across research groups.",
    "Federated learning allows AI models to be trained across distributed devices without centralising data.",
    "Synthetic data generation is used to augment training datasets when real-world data is scarce.",
    "AI governance involves the policies, standards, and accountability mechanisms that guide AI deployment.",
    "Multimodal AI systems can process and generate content across text, image, audio, and video modalities.",
    "The energy consumption of large AI models has raised concerns about their environmental sustainability.",

    # ── Machine Learning (20) ─────────────────────────────────────────────────
    "Machine learning algorithms are widely applied in recommendation systems, fraud detection, and predictive analytics.",
    "Supervised learning trains models on labelled data to make accurate predictions on unseen inputs.",
    "Unsupervised learning discovers hidden patterns in data without the use of labelled examples.",
    "Reinforcement learning trains agents to maximise cumulative reward through trial-and-error interaction.",
    "Deep learning uses multi-layer neural networks to learn hierarchical representations of data.",
    "Gradient descent is the core optimisation algorithm used to train most machine learning models.",
    "Overfitting occurs when a model learns noise in the training data rather than the underlying pattern.",
    "Regularisation techniques such as dropout and L2 penalty help reduce overfitting in neural networks.",
    "Transfer learning allows pre-trained models to be fine-tuned on new tasks with limited labelled data.",
    "Ensemble methods combine multiple weak learners to produce a more accurate and robust prediction.",
    "Cross-validation is used to estimate model generalisation performance on unseen data.",
    "Hyperparameter tuning involves searching for the optimal model configuration using techniques like grid search.",
    "Feature engineering transforms raw data into informative representations that improve model accuracy.",
    "Active learning selects the most informative unlabelled examples for human annotation to minimise labelling cost.",
    "Semi-supervised learning leverages both labelled and unlabelled data to improve model performance.",
    "Bayesian optimisation is an efficient strategy for hyperparameter search in expensive-to-evaluate models.",
    "Causal inference methods help distinguish correlation from causation in machine learning pipelines.",
    "Anomaly detection algorithms identify rare events or observations that deviate significantly from expected patterns.",
    "Contrastive learning trains representations by contrasting similar and dissimilar pairs of examples.",
    "Few-shot learning enables models to generalise to new tasks from very few labelled training examples.",

    # ── NLP (20) ──────────────────────────────────────────────────────────────
    "Natural language processing enables machines to understand, interpret, and generate human language.",
    "Transformer models such as BERT and GPT have revolutionised the field of natural language processing.",
    "Tokenisation converts raw text into numerical tokens that can be processed by language models.",
    "Attention mechanisms allow models to focus on the most relevant parts of the input sequence.",
    "Named entity recognition identifies and classifies proper nouns in text into predefined categories.",
    "Sentiment analysis determines the emotional tone expressed in a piece of text.",
    "Text summarisation condenses long documents while preserving the most important information.",
    "Machine translation systems automatically convert text from one human language to another.",
    "Language models are trained on large corpora to predict the probability of word sequences.",
    "Word embeddings represent words as dense vectors in a high-dimensional semantic space.",
    "Coreference resolution links pronouns and noun phrases in text to the entities they refer to.",
    "Question answering systems retrieve or generate accurate responses to natural language queries.",
    "Dependency parsing identifies the grammatical structure and relationships between words in a sentence.",
    "Text classification assigns predefined categories to documents based on their content.",
    "Dialogue systems model multi-turn conversations to provide coherent and contextually appropriate responses.",
    "Information extraction converts unstructured text into structured data by identifying key entities and relations.",
    "Semantic role labelling identifies the predicate-argument structure of sentences in natural language.",
    "Reading comprehension benchmarks evaluate whether models can answer questions about a given passage.",
    "Language model fine-tuning adapts a pre-trained model to a specific downstream task using domain data.",
    "Prompt engineering designs input templates that guide language models toward desired output behaviour.",

    # ── Plagiarism & Academic Integrity (20) ──────────────────────────────────
    "Plagiarism detection systems compare submitted texts with a reference corpus to identify similarities.",
    "Semantic similarity measures how closely two texts convey the same meaning regardless of wording.",
    "Cosine similarity between sentence embeddings is an effective metric for detecting paraphrased plagiarism.",
    "Students often paraphrase source material to avoid detection by surface-level string matching tools.",
    "Academic integrity policies require all submitted work to be original and properly attributed.",
    "Cross-lingual plagiarism involves translating source material to obscure its origin.",
    "Paraphrase detection is more challenging than verbatim copy detection due to lexical variation.",
    "Citation analysis can reveal undisclosed borrowing even when the text has been substantially reworded.",
    "Automated plagiarism checkers cannot replace human judgment in complex academic misconduct cases.",
    "FAISS enables fast approximate nearest-neighbour search over large embedding databases.",
    "Mosaic plagiarism assembles phrases from multiple sources to create a text that appears original.",
    "Contract cheating involves submitting work produced by a third party as one's own academic output.",
    "Self-plagiarism occurs when an author reuses substantial portions of their own previously published work.",
    "Ghost-writing services that produce academic work for students undermine educational assessment integrity.",
    "Metadata analysis of submitted documents can sometimes reveal inconsistencies suggesting authorship fraud.",
    "Stylometric analysis compares writing style features to identify likely authorship of anonymous texts.",
    "Proper citation practices attribute ideas and quotations to their original authors in academic writing.",
    "Turnitin and similar tools use large databases of academic and web content to detect copied material.",
    "Institutional plagiarism policies define the consequences of academic dishonesty at various severity levels.",
    "Peer review processes in journals help identify undisclosed overlap with previously published research.",

    # ── Healthcare & AI (20) ──────────────────────────────────────────────────
    "Artificial intelligence improves healthcare systems by supporting diagnosis and treatment planning.",
    "Clinical decision support systems use AI to assist physicians in diagnosing complex conditions.",
    "Medical imaging analysis powered by deep learning can detect tumours with high accuracy.",
    "Electronic health records provide large datasets that train predictive models for patient outcomes.",
    "AI-driven drug discovery accelerates the identification of potential therapeutic compounds.",
    "Personalised medicine uses genetic and clinical data to tailor treatments to individual patients.",
    "Remote patient monitoring via wearable sensors generates continuous health data for AI analysis.",
    "Natural language processing extracts structured information from unstructured clinical notes.",
    "Federated learning allows hospitals to collaboratively train models without sharing patient data.",
    "Bias in medical AI systems can lead to unequal treatment outcomes across demographic groups.",
    "Telemedicine platforms use AI to triage patients and route them to appropriate care pathways.",
    "Genomic sequencing data is analysed with machine learning to identify disease-associated genetic variants.",
    "AI models trained on histopathology slides can classify cancer subtypes with expert-level performance.",
    "Predictive models for hospital readmission help allocate post-discharge resources more effectively.",
    "Robotic surgery systems use computer vision and AI to assist surgeons with precision procedures.",
    "Mental health applications use conversational AI to deliver cognitive behavioural therapy techniques.",
    "Sepsis early-warning systems use real-time vital sign monitoring and machine learning to alert clinicians.",
    "AI-powered radiology tools reduce reporting time and flag urgent findings for priority review.",
    "Clinical trial design is being optimised using machine learning to identify eligible patient populations.",
    "Digital pathology leverages AI to automate the analysis of tissue samples at scale.",

    # ── Deep Learning & Architectures (20) ────────────────────────────────────
    "Convolutional neural networks are highly effective for image recognition and object detection tasks.",
    "Recurrent neural networks process sequential data by maintaining an internal hidden state.",
    "The transformer architecture relies entirely on self-attention and has replaced RNNs in many tasks.",
    "BERT is pre-trained using masked language modelling and next-sentence prediction objectives.",
    "GPT models are autoregressive language models trained to predict the next token in a sequence.",
    "Encoder-decoder architectures are commonly used in sequence-to-sequence tasks such as translation.",
    "Batch normalisation stabilises training by normalising layer inputs across each mini-batch.",
    "Residual connections allow gradients to flow through very deep networks without vanishing.",
    "The attention mechanism computes a weighted sum of values based on query-key similarity scores.",
    "Large language models are trained on trillions of tokens and exhibit emergent reasoning capabilities.",
    "Graph neural networks extend deep learning to non-Euclidean data structures such as molecular graphs.",
    "Diffusion models generate high-quality images by learning to reverse a gradual noising process.",
    "Vision transformers apply the self-attention mechanism to image patches for visual recognition tasks.",
    "Neural architecture search automates the design of deep learning model structures using optimisation.",
    "Mixture-of-experts models activate only a subset of parameters per input, improving efficiency at scale.",
    "Contrastive language-image pre-training aligns visual and textual representations in a shared embedding space.",
    "Knowledge distillation transfers learned representations from a large teacher model to a smaller student model.",
    "Sparse attention mechanisms reduce the quadratic complexity of full self-attention in long sequences.",
    "State space models offer an alternative to transformers for efficiently modelling long-range dependencies.",
    "Quantisation reduces model size and inference latency by representing weights with lower precision.",

    # ── General Science & Technology (20) ─────────────────────────────────────
    "Scientific progress depends on reproducible experiments, peer review, and transparent reporting of results.",
    "Quantum computing uses quantum mechanical phenomena to perform computations beyond classical capabilities.",
    "The internet of things connects physical devices to digital networks, enabling data collection at scale.",
    "Cybersecurity involves protecting computer systems and networks from unauthorised access and attack.",
    "Cloud computing delivers on-demand access to computing resources over the internet on a pay-per-use basis.",
    "Blockchain technology provides a decentralised and tamper-resistant ledger for recording transactions.",
    "Robotics combines mechanical engineering, electronics, and software to build machines that perform physical tasks.",
    "Edge computing moves data processing closer to the source to reduce latency and bandwidth usage.",
    "Open-source software allows developers to inspect, modify, and distribute code freely under permissive licences.",
    "5G networks offer significantly higher bandwidth and lower latency than previous mobile communication standards.",
    "Augmented reality overlays digital information on the physical world using cameras and display technology.",
    "Semiconductor fabrication advances have enabled exponential growth in computing power over recent decades.",
    "Digital twins create virtual replicas of physical systems for simulation, monitoring, and optimisation.",
    "Bioinformatics applies computational methods to analyse biological data such as DNA sequences and protein structures.",
    "Space technology development drives innovation in materials science, communication, and remote sensing.",
    "Human-computer interaction research investigates how people engage with digital systems and interfaces.",
    "Data engineering builds pipelines that collect, transform, and store large volumes of structured and unstructured data.",
    "Software testing ensures that code behaves correctly under expected and unexpected conditions.",
    "Version control systems track changes to source code, enabling collaboration and rollback of errors.",
    "API design principles govern how software components expose functionality to other applications.",

    # ── Climate & Environment (20) ────────────────────────────────────────────
    "Climate change is driven primarily by greenhouse gas emissions from human industrial and agricultural activity.",
    "Rising global temperatures are causing more frequent and severe extreme weather events worldwide.",
    "Renewable energy sources such as solar and wind power are essential for reducing carbon emissions.",
    "Carbon capture and storage technologies aim to remove CO2 from the atmosphere or emission sources.",
    "Biodiversity loss threatens ecosystem stability and the services that natural environments provide to humanity.",
    "Deforestation contributes to climate change by reducing the carbon-absorbing capacity of forests.",
    "Ocean acidification caused by CO2 absorption endangers marine ecosystems and coral reef health.",
    "Sustainable agriculture practices aim to produce food while minimising environmental degradation.",
    "The Paris Agreement sets international targets for limiting global warming to well below two degrees Celsius.",
    "Environmental impact assessments evaluate the potential consequences of development projects on ecosystems.",
    "Circular economy models aim to eliminate waste by reusing and recycling materials within production systems.",
    "Water scarcity affects billions of people globally and is projected to worsen with climate change.",
    "Green infrastructure uses natural systems such as wetlands and urban forests to manage environmental challenges.",
    "Life cycle assessment quantifies the environmental impact of a product from production to disposal.",
    "Climate modelling uses atmospheric and oceanic data to simulate future environmental conditions.",
    "Pollution from plastic waste degrades terrestrial and aquatic ecosystems across the globe.",
    "Energy efficiency improvements reduce consumption and emissions without sacrificing economic output.",
    "Environmental justice addresses the disproportionate impact of pollution on marginalised communities.",
    "Methane emissions from livestock and landfills are a significant contributor to global warming.",
    "Ecosystem restoration projects aim to rehabilitate degraded habitats and recover lost biodiversity.",

    # ── Economics & Finance (20) ──────────────────────────────────────────────
    "Monetary policy uses interest rates and money supply to control inflation and stabilise the economy.",
    "Fiscal policy involves government spending and taxation decisions that influence economic activity.",
    "Supply and demand dynamics determine the price of goods and services in competitive markets.",
    "Inflation erodes purchasing power when the general price level rises faster than wages.",
    "Financial markets allocate capital by matching investors with businesses and governments that need funding.",
    "Portfolio diversification reduces investment risk by spreading capital across different asset classes.",
    "Central banks act as lenders of last resort and regulate commercial bank behaviour to maintain stability.",
    "Gross domestic product measures the total economic output of a country over a given period.",
    "Microeconomics studies individual and firm-level decision-making in response to incentives and constraints.",
    "Macroeconomics examines aggregate economic phenomena such as growth, unemployment, and trade balances.",
    "Cryptocurrency markets are highly volatile and largely unregulated compared to traditional financial assets.",
    "Foreign exchange rates reflect the relative value of currencies and influence international trade flows.",
    "Behavioural economics incorporates psychological insights into models of human economic decision-making.",
    "Income inequality has widened in many countries as returns to capital have outpaced wage growth.",
    "Trade agreements reduce tariffs and other barriers to facilitate the exchange of goods between countries.",
    "Financial regulation aims to prevent systemic risk and protect consumers from predatory practices.",
    "Venture capital funds early-stage companies in exchange for equity stakes and potential high returns.",
    "Debt restructuring allows borrowers in financial distress to renegotiate loan terms with their creditors.",
    "Economic recession is characterised by two consecutive quarters of negative GDP growth.",
    "Public goods are non-excludable and non-rivalrous, making them prone to under-provision by private markets.",

    # ── Law & Ethics (20) ────────────────────────────────────────────────────
    "The rule of law requires that all individuals and institutions are subject to and accountable under the law.",
    "Intellectual property law grants creators exclusive rights to their works for a defined period.",
    "Data protection legislation regulates the collection, storage, and use of personal information by organisations.",
    "Due process guarantees that legal proceedings are conducted fairly and in accordance with established rules.",
    "Human rights frameworks protect fundamental freedoms and dignities that are inalienable to all persons.",
    "Contract law governs the formation and enforcement of legally binding agreements between parties.",
    "Tort law provides remedies for individuals who have suffered harm due to the wrongful acts of others.",
    "Corporate governance defines the structures and processes by which companies are directed and controlled.",
    "Antitrust regulation prevents monopolistic behaviour and promotes competition in commercial markets.",
    "The presumption of innocence requires that defendants be considered not guilty until proven otherwise.",
    "Privacy rights protect individuals from unwarranted intrusion into their personal lives and communications.",
    "Environmental law establishes legal obligations for minimising harm to natural ecosystems and resources.",
    "Whistleblower protections safeguard individuals who report illegal or unethical conduct within organisations.",
    "International humanitarian law governs the conduct of armed conflict and protects civilian populations.",
    "Ethical frameworks such as utilitarianism and deontology provide systematic approaches to moral reasoning.",
    "Professional codes of conduct set standards of behaviour for practitioners in regulated fields.",
    "Constitutional law defines the fundamental principles and powers that govern a nation's legal system.",
    "Restorative justice approaches focus on repairing harm through dialogue between offenders and victims.",
    "Immigration law regulates the entry, residence, and status of foreign nationals within a country.",
    "Consumer protection law ensures that businesses deal honestly and fairly with the people they serve.",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text


def _split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 15]


# ══════════════════════════════════════════════════════════════════════════════
# CHECKER CLASS
# ══════════════════════════════════════════════════════════════════════════════
class PlagiarismChecker:
    def __init__(
        self,
        corpus: list = None,
        threshold_high: float = 0.80,
        threshold_moderate: float = 0.60,
        sigmoid_steepness: float = 3.0,
    ):
        """
        Parameters
        ----------
        corpus             : list of reference sentences (defaults to DEFAULT_CORPUS)
        threshold_high     : normalised score ≥ this → High Risk   (default 0.80)
        threshold_moderate : normalised score ≥ this → Moderate Risk (default 0.60)
        sigmoid_steepness  : k in sigmoid 1/(1+exp(-k*z)); higher = steeper curve
        """
        self.corpus              = corpus if corpus is not None else DEFAULT_CORPUS
        self.threshold_high      = threshold_high
        self.threshold_moderate  = threshold_moderate
        self.sigmoid_steepness   = sigmoid_steepness

        self._index       = None
        self._built       = False
        self._corpus_mean = 0.0
        self._corpus_std  = 1.0

    # ── lazy index build ──────────────────────────────────────────────────────
    def _ensure_index(self):
        if self._built:
            return

        embedder  = get_embedder()
        processed = [_preprocess(s) for s in self.corpus]
        embs      = embedder.encode(
            processed, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        self._index = faiss.IndexFlatIP(embs.shape[1])
        self._index.add(embs)

        # Background distribution: nearest-neighbour similarity between corpus sentences
        n = len(self.corpus)
        scores_mat, _ = self._index.search(embs, min(3, n))
        bg_scores = scores_mat[:, 1].tolist() if scores_mat.shape[1] > 1 else []

        if len(bg_scores) >= 4:
            self._corpus_mean = float(np.mean(bg_scores))
            self._corpus_std  = float(np.std(bg_scores)) or 0.01
        else:
            self._corpus_mean = 0.75
            self._corpus_std  = 0.08

        self._built = True

    # ── sigmoid normalisation ─────────────────────────────────────────────────
    def _normalise(self, raw_score: float) -> float:
        """
        Map raw cosine similarity to domain-adjusted [0, 1] via sigmoid.

        z = (raw - corpus_mean) / corpus_std
        normalised = sigmoid(k * z)  where k = sigmoid_steepness

        Properties:
          • Stable regardless of corpus_std magnitude (no linear blow-up)
          • Compresses extreme values smoothly
          • raw == corpus_mean  →  normalised ≈ 0.50
          • raw >> corpus_mean  →  normalised → 1.0
          • raw << corpus_mean  →  normalised → 0.0
        """
        z = (raw_score - self._corpus_mean) / (self._corpus_std + 1e-8)
        return float(1.0 / (1.0 + np.exp(-self.sigmoid_steepness * z)))

    # ── risk classification (on NORMALISED scores) ────────────────────────────
    def classify(self, norm_score: float) -> str:
        if norm_score >= self.threshold_high:
            return "🔴 High Risk"
        elif norm_score >= self.threshold_moderate:
            return "🟠 Moderate Risk"
        return "🟢 Low Risk"

    # ── single embedding lookup ───────────────────────────────────────────────
    def _search_one(self, text: str, top_k: int) -> list:
        embedder = get_embedder()
        q_emb    = embedder.encode(
            [_preprocess(text)], normalize_embeddings=True
        ).astype(np.float32)
        safe_k           = min(top_k, len(self.corpus))
        scores, indices  = self._index.search(q_emb, safe_k)
        results = []
        for raw_score, idx in zip(scores[0], indices[0]):
            raw  = float(np.clip(raw_score, 0.0, 1.0))
            norm = self._normalise(raw)
            results.append({
                "matched_text":     self.corpus[idx],
                "similarity_score": round(norm, 4),
                "raw_score":        round(raw,  4),
                "risk_level":       self.classify(norm),
            })
        return results

    # ── public check ──────────────────────────────────────────────────────────
    def check(self, text: str, top_k: int = 3) -> dict:
        """
        Check *text* for plagiarism.

        Returns
        -------
        dict with keys:
          - matches        : list of top-k results with normalised similarity
          - overall_risk   : document-level risk label
          - max_similarity : highest normalised similarity score found
        """
        self._ensure_index()

        sentences = _split_sentences(text)

        # Short / single-sentence → direct lookup
        if len(sentences) <= 1:
            matches  = self._search_one(text, top_k)
            max_sim  = max(m["similarity_score"] for m in matches)
            return {
                "matches":        matches,
                "overall_risk":   self.classify(max_sim),
                "max_similarity": round(max_sim, 4),
            }

        # Long text → per-sentence best match
        per_sentence = []
        for sent in sentences:
            best = self._search_one(sent, top_k=1)[0]
            per_sentence.append({
                "query_sentence":   sent,
                "matched_text":     best["matched_text"],
                "similarity_score": best["similarity_score"],
                "risk_level":       best["risk_level"],
            })

        per_sentence.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_matches = per_sentence[:top_k]

        scores      = [m["similarity_score"] for m in top_matches]
        max_sim     = scores[0]
        mean_sim    = float(np.mean(scores))
        median_sim  = float(np.median(scores))

        # Improved aggregation: max + mean + median for outlier robustness
        agg_sim = round(
            0.50 * max_sim + 0.30 * mean_sim + 0.20 * median_sim, 4
        )

        return {
            "matches":        top_matches,
            "overall_risk":   self.classify(agg_sim),
            "max_similarity": round(max_sim, 4),
        }


# ── module-level singleton ────────────────────────────────────────────────────
_checker = PlagiarismChecker()


def check_plagiarism(text: str, top_k: int = 3) -> dict:
    """Convenience wrapper. Returns the full result dict from PlagiarismChecker."""
    return _checker.check(text, top_k=top_k)